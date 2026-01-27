//! Deterministic mock exchange for live ingestion + execution testing.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, mpsc, oneshot, Mutex};
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message};

use crate::config::Config;
use crate::types::{OrderIntent, OrderPurpose, Side, TimeInForce, TimestampMs};

use super::orderbook_l2::{BookLevel, OrderBookL2};
use super::runner::{LiveAccountRequest, LiveOrderRequest};
use super::types::{
    AccountEvent, AccountSnapshot, BalanceSnapshot, ExecutionEvent, LiquidationSnapshot,
    MarginSnapshot, MarketDataEvent, PositionSnapshot,
};

#[derive(Debug, Clone)]
pub struct MockExchangeConfig {
    pub levels: usize,
    pub level_size: f64,
    pub mid_price: f64,
    pub spread: f64,
    pub tick_size: f64,
    pub taker_fee_bps: f64,
    pub account_balance_usd: f64,
    pub account_used_usd: f64,
    pub account_available_usd: f64,
    pub account_liq_price_offset: f64,
    pub account_funding_8h: f64,
}

impl Default for MockExchangeConfig {
    fn default() -> Self {
        Self {
            levels: 5,
            level_size: 5.0,
            mid_price: 100.0,
            spread: 1.0,
            tick_size: 0.5,
            taker_fee_bps: 5.0,
            account_balance_usd: 10_000.0,
            account_used_usd: 0.0,
            account_available_usd: 10_000.0,
            account_liq_price_offset: -10.0,
            account_funding_8h: 0.0,
        }
    }
}

#[derive(Debug)]
pub enum MockExchangeCommand {
    Intent(OrderIntent),
    CancelAll {
        venue_index: Option<usize>,
        now_ms: TimestampMs,
        response: oneshot::Sender<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct MockExchangeHandle {
    pub ws_url: String,
    pub market_tx: broadcast::Sender<MarketDataEvent>,
    pub account_tx: broadcast::Sender<AccountEvent>,
    pub account_reconcile_tx: mpsc::Sender<LiveAccountRequest>,
    pub order_tx: mpsc::Sender<LiveOrderRequest>,
    pub command_tx: mpsc::Sender<MockExchangeCommand>,
    pub tick_tx: mpsc::Sender<TimestampMs>,
    pub execution_log: Arc<Mutex<Vec<ExecutionEvent>>>,
    pub open_orders: Arc<Mutex<HashMap<String, MockOpenOrder>>>,
    pub positions: Arc<Mutex<Vec<f64>>>,
}

impl MockExchangeHandle {
    pub async fn open_orders_len(&self) -> usize {
        self.open_orders.lock().await.len()
    }

    pub async fn position(&self, venue_index: usize) -> f64 {
        self.positions
            .lock()
            .await
            .get(venue_index)
            .copied()
            .unwrap_or(0.0)
    }

    pub async fn snapshot_open_orders(
        &self,
        venue_index: usize,
        venue_id: String,
        now_ms: TimestampMs,
        seq: u64,
    ) -> super::types::OrderSnapshot {
        let open = self.open_orders.lock().await;
        let mut orders = Vec::new();
        for (order_id, order) in open.iter() {
            if order.venue_index != venue_index {
                continue;
            }
            orders.push(super::types::OpenOrderSnapshot {
                order_id: order_id.clone(),
                side: order.side,
                price: order.price,
                size: order.size,
                purpose: order.purpose,
            });
        }
        super::types::OrderSnapshot {
            venue_index,
            venue_id,
            seq,
            timestamp_ms: now_ms,
            open_orders: orders,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MockOpenOrder {
    pub venue_index: usize,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    pub time_in_force: TimeInForce,
    pub post_only: bool,
    pub reduce_only: bool,
}

#[derive(Debug)]
struct VenueBook {
    book: OrderBookL2,
    market_seq: u64,
    exec_seq: u64,
}

pub async fn spawn_mock_exchange(cfg: &Config, mock_cfg: MockExchangeConfig) -> MockExchangeHandle {
    let (market_tx, _) = broadcast::channel::<MarketDataEvent>(1024);
    let (account_tx, _) = broadcast::channel::<AccountEvent>(1024);
    let (order_tx, order_rx) = mpsc::channel::<LiveOrderRequest>(1024);
    let (account_request_tx, account_request_rx) = mpsc::channel::<LiveAccountRequest>(1024);
    let (command_tx, command_rx) = mpsc::channel::<MockExchangeCommand>(1024);
    let (tick_tx, tick_rx) = mpsc::channel::<TimestampMs>(1024);
    let execution_log = Arc::new(Mutex::new(Vec::new()));
    let open_orders = Arc::new(Mutex::new(HashMap::new()));
    let positions = Arc::new(Mutex::new(vec![0.0_f64; cfg.venues.len()]));

    let ws_addr = spawn_ws_server(market_tx.clone()).await;
    let ws_url = format!("ws://{}", ws_addr);

    let venues = cfg
        .venues
        .iter()
        .enumerate()
        .map(|(_idx, _)| VenueBook {
            book: OrderBookL2::new(),
            market_seq: 0,
            exec_seq: 0,
        })
        .collect::<Vec<_>>();

    let handle = MockExchangeHandle {
        ws_url,
        market_tx: market_tx.clone(),
        account_tx: account_tx.clone(),
        account_reconcile_tx: account_request_tx,
        order_tx,
        command_tx,
        tick_tx,
        execution_log: execution_log.clone(),
        open_orders: open_orders.clone(),
        positions: positions.clone(),
    };

    tokio::spawn(run_exchange(
        cfg.clone(),
        mock_cfg,
        market_tx,
        account_tx,
        order_rx,
        account_request_rx,
        command_rx,
        tick_rx,
        venues,
        execution_log,
        open_orders,
        positions,
    ));

    handle
}

pub async fn spawn_market_ws_client(ws_url: &str) -> mpsc::Receiver<MarketDataEvent> {
    let (tx, rx) = mpsc::channel(1024);
    let ws_url = ws_url.to_string();
    tokio::spawn(async move {
        if let Ok((ws_stream, _)) = connect_async(ws_url).await {
            let (_, mut reader) = ws_stream.split();
            while let Some(Ok(msg)) = reader.next().await {
                if let Message::Text(text) = msg {
                    if let Ok(event) = serde_json::from_str::<MarketDataEvent>(&text) {
                        let _ = tx.send(event).await;
                    }
                }
            }
        }
    });
    rx
}

async fn spawn_ws_server(market_tx: broadcast::Sender<MarketDataEvent>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind ws");
    let addr = listener.local_addr().expect("ws addr");
    tokio::spawn(async move {
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(v) => v,
                Err(_) => continue,
            };
            let mut market_rx = market_tx.subscribe();
            tokio::spawn(async move {
                if let Ok(ws) = accept_async(stream).await {
                    let (mut writer, _) = ws.split();
                    while let Ok(event) = market_rx.recv().await {
                        if let Ok(payload) = serde_json::to_string(&event) {
                            if writer.send(Message::Text(payload)).await.is_err() {
                                break;
                            }
                        }
                    }
                }
            });
        }
    });
    addr
}

async fn run_exchange(
    cfg: Config,
    mock_cfg: MockExchangeConfig,
    market_tx: broadcast::Sender<MarketDataEvent>,
    account_tx: broadcast::Sender<AccountEvent>,
    mut order_rx: mpsc::Receiver<LiveOrderRequest>,
    mut account_request_rx: mpsc::Receiver<LiveAccountRequest>,
    mut command_rx: mpsc::Receiver<MockExchangeCommand>,
    mut tick_rx: mpsc::Receiver<TimestampMs>,
    mut venues: Vec<VenueBook>,
    execution_log: Arc<Mutex<Vec<ExecutionEvent>>>,
    open_orders: Arc<Mutex<HashMap<String, MockOpenOrder>>>,
    positions: Arc<Mutex<Vec<f64>>>,
) {
    let mut tick_idx: u64 = 0;
    loop {
        tokio::select! {
            Some(now_ms) = tick_rx.recv() => {
                for (idx, venue) in venues.iter_mut().enumerate() {
                    let mid = mock_cfg.mid_price + (idx as f64 * 0.5) + mock_cfg.tick_size * (tick_idx as f64 % 2.0);
                    let bids = build_levels(mid, mock_cfg.spread, mock_cfg.tick_size, mock_cfg.level_size, mock_cfg.levels, true);
                    let asks = build_levels(mid, mock_cfg.spread, mock_cfg.tick_size, mock_cfg.level_size, mock_cfg.levels, false);

                    venue.market_seq += 1;
                    let snapshot = super::types::L2Snapshot {
                        venue_index: idx,
                        venue_id: cfg.venues[idx].id.to_string(),
                        seq: venue.market_seq,
                        timestamp_ms: now_ms,
                        bids,
                        asks,
                    };
                    let _ = venue.book.apply_snapshot(&snapshot.bids, &snapshot.asks, snapshot.seq);
                    let _ = market_tx.send(MarketDataEvent::L2Snapshot(snapshot));

                    let account_snapshot = build_account_snapshot(
                        &cfg,
                        &mock_cfg,
                        &positions,
                        idx,
                        now_ms,
                        venue.market_seq,
                        mid,
                    )
                    .await;
                    let _ = account_tx.send(AccountEvent::Snapshot(account_snapshot));
                }
                tick_idx += 1;
            }
            Some(req) = account_request_rx.recv() => {
                let venue_index = req
                    .venue_index
                    .unwrap_or(0)
                    .min(cfg.venues.len().saturating_sub(1));
                let mid = venues
                    .get(venue_index)
                    .and_then(|venue| venue.book.best_bid().map(|b| b.price))
                    .unwrap_or(mock_cfg.mid_price);
                let snapshot =
                    build_account_snapshot(&cfg, &mock_cfg, &positions, venue_index, req.now_ms, 0, mid)
                        .await;
                let _ = req.response.send(snapshot);
            }
            Some(req) = order_rx.recv() => {
                let events =
                    handle_order_request(
                        req.intents,
                        req.now_ms,
                        &mock_cfg,
                        &mut venues,
                        &execution_log,
                        &open_orders,
                        &positions,
                    )
                        .await;
                let _ = req.response.send(events);
            }
            Some(cmd) = command_rx.recv() => {
                handle_command(cmd, &mut venues, &execution_log, &open_orders).await;
            }
            else => break,
        }
    }
}

async fn handle_command(
    cmd: MockExchangeCommand,
    venues: &mut [VenueBook],
    execution_log: &Arc<Mutex<Vec<ExecutionEvent>>>,
    open_orders: &Arc<Mutex<HashMap<String, MockOpenOrder>>>,
) {
    match cmd {
        MockExchangeCommand::CancelAll {
            venue_index,
            now_ms,
            response,
        } => {
            let cancelled = {
                let mut open = open_orders.lock().await;
                let mut cancelled = Vec::new();
                for (order_id, order) in open.iter() {
                    if venue_index.map_or(true, |idx| idx == order.venue_index) {
                        cancelled.push(order_id.clone());
                    }
                }
                for order_id in &cancelled {
                    open.remove(order_id);
                }
                cancelled
            };
            let count = cancelled.len();
            let venue_idx = venue_index.unwrap_or(usize::MAX);
            let venue_opt = if venue_idx < venues.len() {
                Some(&mut venues[venue_idx])
            } else {
                venues.first_mut()
            };
            if let Some(venue) = venue_opt {
                venue.exec_seq += 1;
                let event = ExecutionEvent::CancelAllAccepted(super::types::CancelAllAccepted {
                    venue_index: venue_idx,
                    venue_id: "all".to_string(),
                    seq: venue.exec_seq,
                    timestamp_ms: now_ms,
                    count,
                });
                execution_log.lock().await.push(event.clone());
            }
            let _ = response.send(count);
        }
        MockExchangeCommand::Intent(_) => {}
    }
}

async fn handle_order_request(
    intents: Vec<OrderIntent>,
    now_ms: TimestampMs,
    mock_cfg: &MockExchangeConfig,
    venues: &mut [VenueBook],
    execution_log: &Arc<Mutex<Vec<ExecutionEvent>>>,
    open_orders: &Arc<Mutex<HashMap<String, MockOpenOrder>>>,
    positions: &Arc<Mutex<Vec<f64>>>,
) -> Vec<ExecutionEvent> {
    let mut events = Vec::new();
    for intent in intents {
        match intent {
            OrderIntent::Place(place) => {
                let place_events = handle_place_intent(
                    place,
                    now_ms,
                    mock_cfg,
                    venues,
                    execution_log,
                    open_orders,
                    positions,
                )
                .await;
                events.extend(place_events);
            }
            OrderIntent::Cancel(cancel) => {
                let cancel_events =
                    handle_cancel_intent(cancel, now_ms, venues, execution_log, open_orders).await;
                events.extend(cancel_events);
            }
            OrderIntent::Replace(replace) => {
                let cancel = crate::types::CancelOrderIntent {
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.clone(),
                    order_id: replace.order_id.clone(),
                };
                let place = crate::types::PlaceOrderIntent {
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.clone(),
                    side: replace.side,
                    price: replace.price,
                    size: replace.size,
                    purpose: replace.purpose,
                    time_in_force: replace.time_in_force,
                    post_only: replace.post_only,
                    reduce_only: replace.reduce_only,
                    client_order_id: replace.client_order_id.clone(),
                };
                let cancel_events =
                    handle_cancel_intent(cancel, now_ms, venues, execution_log, open_orders).await;
                let place_events = handle_place_intent(
                    place,
                    now_ms,
                    mock_cfg,
                    venues,
                    execution_log,
                    open_orders,
                    positions,
                )
                .await;
                events.extend(cancel_events);
                events.extend(place_events);
            }
            OrderIntent::CancelAll(cancel_all) => {
                let venue_index = cancel_all.venue_index;
                let cmd = MockExchangeCommand::CancelAll {
                    venue_index,
                    now_ms,
                    response: tokio::sync::oneshot::channel().0,
                };
                handle_command(cmd, venues, execution_log, open_orders).await;
            }
        }
    }
    events
}

async fn handle_place_intent(
    place: crate::types::PlaceOrderIntent,
    now_ms: TimestampMs,
    mock_cfg: &MockExchangeConfig,
    venues: &mut [VenueBook],
    execution_log: &Arc<Mutex<Vec<ExecutionEvent>>>,
    open_orders: &Arc<Mutex<HashMap<String, MockOpenOrder>>>,
    positions: &Arc<Mutex<Vec<f64>>>,
) -> Vec<ExecutionEvent> {
    let mut events = Vec::new();
    let venue_index = place.venue_index;
    let Some(venue) = venues.get_mut(venue_index) else {
        return events;
    };
    if place.size <= 0.0 || !place.price.is_finite() {
        venue.exec_seq += 1;
        let event = ExecutionEvent::OrderRejected(super::types::OrderRejected {
            venue_index,
            venue_id: place.venue_id.to_string(),
            seq: venue.exec_seq,
            timestamp_ms: now_ms,
            order_id: place.client_order_id.clone(),
            reason: "Invalid price/size".to_string(),
        });
        record_event(&event, &execution_log).await;
        events.push(event);
        return events;
    }

    let mut allowed_size = place.size;
    let is_post_only = place.post_only;
    let (best_bid, best_ask) = (
        venue.book.best_bid().map(|l| l.price),
        venue.book.best_ask().map(|l| l.price),
    );
    if is_post_only && crosses_book(place.side, place.price, best_bid, best_ask) {
        venue.exec_seq += 1;
        let event = ExecutionEvent::OrderRejected(super::types::OrderRejected {
            venue_index,
            venue_id: place.venue_id.to_string(),
            seq: venue.exec_seq,
            timestamp_ms: now_ms,
            order_id: place.client_order_id.clone(),
            reason: "Post-only would cross".to_string(),
        });
        record_event(&event, &execution_log).await;
        events.push(event);
        return events;
    }

    if place.reduce_only {
        let pos = positions
            .lock()
            .await
            .get(place.venue_index)
            .copied()
            .unwrap_or(0.0);
        allowed_size = match place.side {
            Side::Buy if pos < 0.0 => pos.abs().min(place.size),
            Side::Sell if pos > 0.0 => pos.abs().min(place.size),
            _ => 0.0,
        };
        if allowed_size <= 0.0 {
            venue.exec_seq += 1;
            let event = ExecutionEvent::OrderRejected(super::types::OrderRejected {
                venue_index,
                venue_id: place.venue_id.to_string(),
                seq: venue.exec_seq,
                timestamp_ms: now_ms,
                order_id: place.client_order_id.clone(),
                reason: "Reduce-only would increase position".to_string(),
            });
            record_event(&event, &execution_log).await;
            events.push(event);
            return events;
        }
    }

    let effective_size = allowed_size;
    let order_id = place
        .client_order_id
        .clone()
        .unwrap_or_else(|| format!("mock_{}_{}", venue_index, venue.exec_seq));
    venue.exec_seq += 1;
    let accepted = ExecutionEvent::OrderAccepted(super::types::OrderAccepted {
        venue_index,
        venue_id: place.venue_id.to_string(),
        seq: venue.exec_seq,
        timestamp_ms: now_ms,
        order_id: order_id.clone(),
        client_order_id: place.client_order_id.clone(),
        side: place.side,
        price: place.price,
        size: effective_size,
        purpose: place.purpose,
    });
    record_event(&accepted, &execution_log).await;
    events.push(accepted.clone());

    if place.time_in_force == TimeInForce::Ioc {
        let (fills, remaining) = match_ioc(place.side, place.price, effective_size, &venue.book);
        for fill in fills {
            if let Some(pos) = positions.lock().await.get_mut(place.venue_index) {
                *pos += match place.side {
                    Side::Buy => fill.size,
                    Side::Sell => -fill.size,
                };
            }
            venue.exec_seq += 1;
            let event = ExecutionEvent::Filled(super::types::Fill {
                venue_index,
                venue_id: place.venue_id.to_string(),
                seq: venue.exec_seq,
                timestamp_ms: now_ms,
                order_id: Some(order_id.clone()),
                client_order_id: place.client_order_id.clone(),
                fill_id: Some(format!("mock_fill_{}_{}", venue_index, venue.exec_seq)),
                side: place.side,
                price: fill.price,
                size: fill.size,
                purpose: place.purpose,
                fee_bps: mock_cfg.taker_fee_bps,
            });
            record_event(&event, &execution_log).await;
            events.push(event);
        }
        if remaining > 0.0 {
            venue.exec_seq += 1;
            let event = ExecutionEvent::CancelAccepted(super::types::CancelAccepted {
                venue_index,
                venue_id: place.venue_id.to_string(),
                seq: venue.exec_seq,
                timestamp_ms: now_ms,
                order_id: order_id.clone(),
            });
            record_event(&event, &execution_log).await;
            events.push(event);
        }
    } else {
        let mut open = open_orders.lock().await;
        open.insert(
            order_id.clone(),
            MockOpenOrder {
                venue_index,
                side: place.side,
                price: place.price,
                size: place.size,
                purpose: place.purpose,
                time_in_force: place.time_in_force,
                post_only: place.post_only,
                reduce_only: place.reduce_only,
            },
        );
    }
    events
}

async fn handle_cancel_intent(
    cancel: crate::types::CancelOrderIntent,
    now_ms: TimestampMs,
    venues: &mut [VenueBook],
    execution_log: &Arc<Mutex<Vec<ExecutionEvent>>>,
    open_orders: &Arc<Mutex<HashMap<String, MockOpenOrder>>>,
) -> Vec<ExecutionEvent> {
    let mut events = Vec::new();
    let venue_index = cancel.venue_index;
    let Some(venue) = venues.get_mut(venue_index) else {
        return events;
    };
    let removed = {
        let mut open = open_orders.lock().await;
        open.remove(&cancel.order_id).is_some()
    };
    if removed {
        venue.exec_seq += 1;
        let event = ExecutionEvent::CancelAccepted(super::types::CancelAccepted {
            venue_index,
            venue_id: cancel.venue_id.to_string(),
            seq: venue.exec_seq,
            timestamp_ms: now_ms,
            order_id: cancel.order_id.clone(),
        });
        record_event(&event, &execution_log).await;
        events.push(event);
    } else {
        venue.exec_seq += 1;
        let event = ExecutionEvent::CancelRejected(super::types::CancelRejected {
            venue_index,
            venue_id: cancel.venue_id.to_string(),
            seq: venue.exec_seq,
            timestamp_ms: now_ms,
            order_id: Some(cancel.order_id.clone()),
            reason: "Unknown order_id".to_string(),
        });
        record_event(&event, &execution_log).await;
        events.push(event);
    }
    events
}

async fn record_event(event: &ExecutionEvent, log: &Arc<Mutex<Vec<ExecutionEvent>>>) {
    log.lock().await.push(event.clone());
}

async fn build_account_snapshot(
    cfg: &Config,
    mock_cfg: &MockExchangeConfig,
    positions: &Arc<Mutex<Vec<f64>>>,
    venue_index: usize,
    now_ms: TimestampMs,
    seq: u64,
    mid: f64,
) -> AccountSnapshot {
    let pos = positions
        .lock()
        .await
        .get(venue_index)
        .copied()
        .unwrap_or(0.0);
    let positions = if pos.abs() > 0.0 {
        vec![PositionSnapshot {
            symbol: "TAO".to_string(),
            size: pos,
            entry_price: mid,
        }]
    } else {
        Vec::new()
    };
    let balances = vec![BalanceSnapshot {
        asset: "USD".to_string(),
        total: mock_cfg.account_balance_usd,
        available: mock_cfg.account_available_usd,
    }];
    let margin = MarginSnapshot {
        balance_usd: mock_cfg.account_balance_usd,
        used_usd: mock_cfg.account_used_usd,
        available_usd: mock_cfg.account_available_usd,
    };
    let liquidation = LiquidationSnapshot {
        price_liq: Some(mid + mock_cfg.account_liq_price_offset),
        dist_liq_sigma: None,
    };
    AccountSnapshot {
        venue_index,
        venue_id: cfg.venues[venue_index].id.clone(),
        seq,
        timestamp_ms: now_ms,
        positions,
        balances,
        funding_8h: Some(mock_cfg.account_funding_8h),
        margin,
        liquidation,
    }
}

fn build_levels(
    mid: f64,
    spread: f64,
    tick_size: f64,
    level_size: f64,
    levels: usize,
    is_bid: bool,
) -> Vec<BookLevel> {
    let mut out = Vec::with_capacity(levels);
    for i in 0..levels {
        let offset = (i as f64) * tick_size;
        let price = if is_bid {
            mid - (spread / 2.0) - offset
        } else {
            mid + (spread / 2.0) + offset
        };
        out.push(BookLevel {
            price,
            size: level_size,
        });
    }
    out
}

fn crosses_book(side: Side, price: f64, best_bid: Option<f64>, best_ask: Option<f64>) -> bool {
    match side {
        Side::Buy => best_ask.map_or(false, |ask| price >= ask),
        Side::Sell => best_bid.map_or(false, |bid| price <= bid),
    }
}

struct MatchedFill {
    price: f64,
    size: f64,
}

fn match_ioc(
    side: Side,
    limit_price: f64,
    size: f64,
    book: &OrderBookL2,
) -> (Vec<MatchedFill>, f64) {
    let mut remaining = size;
    let mut fills = Vec::new();
    let levels: Vec<BookLevel> = match side {
        Side::Buy => book.asks().to_vec(),
        Side::Sell => book.bids().to_vec(),
    };
    for level in levels {
        if remaining <= 0.0 {
            break;
        }
        let price_ok = match side {
            Side::Buy => level.price <= limit_price,
            Side::Sell => level.price >= limit_price,
        };
        if !price_ok {
            break;
        }
        let fill_size = remaining.min(level.size);
        if fill_size <= 0.0 {
            continue;
        }
        fills.push(MatchedFill {
            price: level.price,
            size: fill_size,
        });
        remaining -= fill_size;
    }
    (fills, remaining)
}
