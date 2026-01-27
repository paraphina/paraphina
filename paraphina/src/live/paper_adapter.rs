//! Deterministic paper execution adapter (feature-gated).

use std::collections::HashMap;

use crate::config::Config;
use crate::types::{OrderIntent, OrderPurpose, Side, TimeInForce, TimestampMs};

use super::instrument::InstrumentSpec;
use super::ops::config_hash;
use super::types::{
    AccountSnapshot, BalanceSnapshot, CancelAccepted, CancelAllAccepted, CancelRejected,
    ExecutionEvent, Fill, LiquidationSnapshot, MarginSnapshot, OpenOrderSnapshot, OrderAccepted,
    OrderRejected, OrderSnapshot, PositionSnapshot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaperFillMode {
    None,
    Marketable,
    Mid,
    Always,
}

impl PaperFillMode {
    pub fn from_env() -> Self {
        match std::env::var("PARAPHINA_PAPER_FILL_MODE")
            .unwrap_or_else(|_| "none".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "marketable" => PaperFillMode::Marketable,
            "mid" => PaperFillMode::Mid,
            "always" => PaperFillMode::Always,
            _ => PaperFillMode::None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PaperExecutionConfig {
    pub fill_mode: PaperFillMode,
    pub slippage_bps: f64,
}

impl PaperExecutionConfig {
    pub fn from_env() -> Self {
        let slippage_bps = std::env::var("PARAPHINA_PAPER_SLIPPAGE_BPS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            .max(0.0);
        Self {
            fill_mode: PaperFillMode::from_env(),
            slippage_bps,
        }
    }
}

#[derive(Debug, Clone)]
struct PaperOrder {
    venue_index: usize,
    venue_id: String,
    side: Side,
    price: f64,
    size: f64,
    purpose: OrderPurpose,
    time_in_force: TimeInForce,
    post_only: bool,
    reduce_only: bool,
    client_order_id: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct PaperMarketUpdate {
    pub venue_index: usize,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub timestamp_ms: TimestampMs,
}

#[derive(Debug)]
pub struct PaperExecutionAdapter {
    cfg_hash: u64,
    nonce: u64,
    seq: u64,
    account_seq: u64,
    open_orders: HashMap<String, PaperOrder>,
    specs: Vec<InstrumentSpec>,
    best_bid_ask: HashMap<usize, (Option<f64>, Option<f64>)>,
    fees: Vec<(f64, f64)>,
    positions: Vec<f64>,
    cash_balance_usd: Vec<f64>,
    touched_accounts: std::collections::BTreeSet<usize>,
    config: PaperExecutionConfig,
}

impl PaperExecutionAdapter {
    pub fn new(cfg: &Config) -> Self {
        let fees = cfg
            .venues
            .iter()
            .map(|v| (v.maker_fee_bps - v.maker_rebate_bps, v.taker_fee_bps))
            .collect();
        let cash_balance = std::env::var("PARAPHINA_PAPER_BALANCE_USD")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(10_000.0);
        Self {
            cfg_hash: config_hash(cfg),
            nonce: 0,
            seq: 0,
            account_seq: 0,
            open_orders: HashMap::new(),
            specs: InstrumentSpec::from_config(cfg),
            best_bid_ask: HashMap::new(),
            fees,
            positions: vec![0.0; cfg.venues.len()],
            cash_balance_usd: vec![cash_balance; cfg.venues.len()],
            touched_accounts: std::collections::BTreeSet::new(),
            config: PaperExecutionConfig::from_env(),
        }
    }

    pub fn config(&self) -> &PaperExecutionConfig {
        &self.config
    }

    pub fn update_best_bid_ask(&mut self, update: PaperMarketUpdate) -> Vec<ExecutionEvent> {
        self.best_bid_ask
            .insert(update.venue_index, (update.best_bid, update.best_ask));
        self.fill_open_orders(update.venue_index, update.timestamp_ms)
    }

    pub fn handle_intents(
        &mut self,
        intents: Vec<OrderIntent>,
        tick: u64,
        now_ms: TimestampMs,
    ) -> Vec<ExecutionEvent> {
        let mut events = Vec::new();
        let mut touched_venues = std::collections::BTreeSet::new();
        for intent in intents {
            match intent {
                OrderIntent::Place(place) => {
                    touched_venues.insert(place.venue_index);
                    let place_events = self.handle_place(place, tick, now_ms);
                    events.extend(place_events);
                }
                OrderIntent::Cancel(cancel) => {
                    touched_venues.insert(cancel.venue_index);
                    let event = self.handle_cancel(cancel, now_ms);
                    events.push(event);
                }
                OrderIntent::Replace(replace) => {
                    touched_venues.insert(replace.venue_index);
                    let cancel_event = self.handle_cancel(
                        crate::types::CancelOrderIntent {
                            venue_index: replace.venue_index,
                            venue_id: replace.venue_id.clone(),
                            order_id: replace.order_id.clone(),
                        },
                        now_ms,
                    );
                    events.push(cancel_event);
                    let place_events = self.handle_place(
                        crate::types::PlaceOrderIntent {
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
                        },
                        tick,
                        now_ms,
                    );
                    events.extend(place_events);
                }
                OrderIntent::CancelAll(cancel_all) => {
                    if let Some(idx) = cancel_all.venue_index {
                        touched_venues.insert(idx);
                    }
                    let event = self.handle_cancel_all(cancel_all, now_ms);
                    events.push(event);
                }
            }
        }
        for venue_index in touched_venues {
            let snapshot = self.snapshot_open_orders(venue_index, now_ms);
            events.push(ExecutionEvent::OrderSnapshot(snapshot));
        }
        events
    }

    fn handle_place(
        &mut self,
        mut place: crate::types::PlaceOrderIntent,
        tick: u64,
        now_ms: TimestampMs,
    ) -> Vec<ExecutionEvent> {
        let mut events = Vec::new();
        if let Some(spec) = self.specs.get(place.venue_index) {
            place.price = spec.round_price(place.price);
            place.size = spec.round_size(place.size);
            if !spec.meets_min_notional(place.size, place.price) {
                self.seq = self.seq.wrapping_add(1);
                events.push(ExecutionEvent::OrderRejected(OrderRejected {
                    venue_index: place.venue_index,
                    venue_id: place.venue_id.to_string(),
                    seq: self.seq,
                    timestamp_ms: now_ms,
                    order_id: place.client_order_id.clone(),
                    reason: "min_notional_usd".to_string(),
                }));
                return events;
            }
        }

        if place.post_only && self.crosses_book(&place) {
            self.seq = self.seq.wrapping_add(1);
            events.push(ExecutionEvent::OrderRejected(OrderRejected {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: place.client_order_id.clone(),
                reason: "post_only_cross".to_string(),
            }));
            return events;
        }

        let order_id = self.next_order_id(
            place.venue_id.as_ref(),
            tick,
            place.side,
            place.purpose,
            place.client_order_id.clone(),
        );
        let client_order_id = Some(order_id.clone());
        self.seq = self.seq.wrapping_add(1);
        events.push(ExecutionEvent::OrderAccepted(OrderAccepted {
            venue_index: place.venue_index,
            venue_id: place.venue_id.to_string(),
            seq: self.seq,
            timestamp_ms: now_ms,
            order_id: order_id.clone(),
            client_order_id: client_order_id.clone(),
            side: place.side,
            price: place.price,
            size: place.size,
            purpose: place.purpose,
        }));

        let fill_decision = self.decide_fill(&place);
        if let Some(fill_price) = fill_decision.fill_price {
            let fee_bps = self.fee_bps(place.venue_index, fill_decision.marketable);
            self.apply_paper_fill(
                place.venue_index,
                place.side,
                place.size,
                fill_price,
                fee_bps,
            );
            self.seq = self.seq.wrapping_add(1);
            events.push(ExecutionEvent::Filled(Fill {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: Some(order_id.clone()),
                client_order_id: client_order_id.clone(),
                fill_id: None,
                side: place.side,
                price: fill_price,
                size: place.size,
                purpose: place.purpose,
                fee_bps,
            }));
            return events;
        }

        if place.time_in_force == TimeInForce::Ioc {
            self.seq = self.seq.wrapping_add(1);
            events.push(ExecutionEvent::CancelAccepted(CancelAccepted {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: order_id.clone(),
            }));
            return events;
        }

        self.open_orders.insert(
            order_id,
            PaperOrder {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                side: place.side,
                price: place.price,
                size: place.size,
                purpose: place.purpose,
                time_in_force: place.time_in_force,
                post_only: place.post_only,
                reduce_only: place.reduce_only,
                client_order_id,
            },
        );
        events
    }

    fn handle_cancel(
        &mut self,
        cancel: crate::types::CancelOrderIntent,
        now_ms: TimestampMs,
    ) -> ExecutionEvent {
        if self.open_orders.remove(&cancel.order_id).is_some() {
            self.seq = self.seq.wrapping_add(1);
            ExecutionEvent::CancelAccepted(CancelAccepted {
                venue_index: cancel.venue_index,
                venue_id: cancel.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: cancel.order_id.clone(),
            })
        } else {
            self.seq = self.seq.wrapping_add(1);
            ExecutionEvent::CancelRejected(CancelRejected {
                venue_index: cancel.venue_index,
                venue_id: cancel.venue_id.to_string(),
                seq: self.seq,
                timestamp_ms: now_ms,
                order_id: Some(cancel.order_id.clone()),
                reason: "order_not_found".to_string(),
            })
        }
    }

    fn handle_cancel_all(
        &mut self,
        cancel_all: crate::types::CancelAllOrderIntent,
        now_ms: TimestampMs,
    ) -> ExecutionEvent {
        let mut cancelled = Vec::new();
        for (order_id, order) in &self.open_orders {
            if cancel_all
                .venue_index
                .map(|idx| idx == order.venue_index)
                .unwrap_or(true)
            {
                cancelled.push(order_id.clone());
            }
        }
        for order_id in &cancelled {
            self.open_orders.remove(order_id);
        }
        let venue_id = cancel_all
            .venue_id
            .as_ref()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "all".to_string());
        self.seq = self.seq.wrapping_add(1);
        ExecutionEvent::CancelAllAccepted(CancelAllAccepted {
            venue_index: cancel_all.venue_index.unwrap_or(0),
            venue_id,
            seq: self.seq,
            timestamp_ms: now_ms,
            count: cancelled.len(),
        })
    }

    fn fill_open_orders(&mut self, venue_index: usize, now_ms: TimestampMs) -> Vec<ExecutionEvent> {
        if self.config.fill_mode == PaperFillMode::None {
            return Vec::new();
        }
        let mut events = Vec::new();
        let order_ids: Vec<String> = self
            .open_orders
            .iter()
            .filter(|(_, order)| order.venue_index == venue_index)
            .map(|(id, _)| id.clone())
            .collect();
        for order_id in order_ids {
            let Some(order) = self.open_orders.get(&order_id).cloned() else {
                continue;
            };
            let fill_decision = self.decide_fill_for_order(&order);
            if let Some(fill_price) = fill_decision.fill_price {
                self.open_orders.remove(&order_id);
                let fee_bps = self.fee_bps(order.venue_index, fill_decision.marketable);
                self.apply_paper_fill(
                    order.venue_index,
                    order.side,
                    order.size,
                    fill_price,
                    fee_bps,
                );
                self.seq = self.seq.wrapping_add(1);
                events.push(ExecutionEvent::Filled(Fill {
                    venue_index: order.venue_index,
                    venue_id: order.venue_id.to_string(),
                    seq: self.seq,
                    timestamp_ms: now_ms,
                    order_id: Some(order_id.clone()),
                    client_order_id: order.client_order_id.clone(),
                    fill_id: None,
                    side: order.side,
                    price: fill_price,
                    size: order.size,
                    purpose: order.purpose,
                    fee_bps,
                }));
            }
        }
        events
    }

    pub fn drain_account_snapshots(
        &mut self,
        venue_id_lookup: &[String],
        now_ms: TimestampMs,
    ) -> Vec<AccountSnapshot> {
        if self.touched_accounts.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let venues: Vec<usize> = self.touched_accounts.iter().copied().collect();
        self.touched_accounts.clear();
        for venue_index in venues {
            let venue_id = venue_id_lookup
                .get(venue_index)
                .cloned()
                .unwrap_or_else(|| "paper".to_string());
            let position = self.positions.get(venue_index).copied().unwrap_or(0.0);
            let cash = self
                .cash_balance_usd
                .get(venue_index)
                .copied()
                .unwrap_or(0.0);
            self.account_seq = self.account_seq.wrapping_add(1);
            out.push(AccountSnapshot {
                venue_index,
                venue_id,
                seq: self.account_seq,
                timestamp_ms: now_ms,
                positions: vec![PositionSnapshot {
                    symbol: "TAO".to_string(),
                    size: position,
                    entry_price: 0.0,
                }],
                balances: vec![BalanceSnapshot {
                    asset: "USD".to_string(),
                    total: cash,
                    available: cash,
                }],
                funding_8h: None,
                margin: MarginSnapshot {
                    balance_usd: cash,
                    used_usd: 0.0,
                    available_usd: cash,
                },
                liquidation: LiquidationSnapshot {
                    price_liq: None,
                    dist_liq_sigma: None,
                },
            });
        }
        out
    }

    fn snapshot_open_orders(&mut self, venue_index: usize, now_ms: TimestampMs) -> OrderSnapshot {
        let mut orders = Vec::new();
        for (order_id, order) in &self.open_orders {
            if order.venue_index != venue_index {
                continue;
            }
            orders.push(OpenOrderSnapshot {
                order_id: order_id.clone(),
                side: order.side,
                price: order.price,
                size: order.size,
                purpose: order.purpose,
            });
        }
        self.seq = self.seq.wrapping_add(1);
        OrderSnapshot {
            venue_index,
            venue_id: self
                .specs
                .get(venue_index)
                .map(|s| s.venue_id.to_string())
                .unwrap_or_else(|| "paper".to_string()),
            seq: self.seq,
            timestamp_ms: now_ms,
            open_orders: orders,
        }
    }

    fn apply_paper_fill(
        &mut self,
        venue_index: usize,
        side: Side,
        size: f64,
        price: f64,
        fee_bps: f64,
    ) {
        let signed = match side {
            Side::Buy => size,
            Side::Sell => -size,
        };
        if let Some(pos) = self.positions.get_mut(venue_index) {
            *pos += signed;
        }
        let fee = price * size * (fee_bps / 10_000.0);
        if let Some(cash) = self.cash_balance_usd.get_mut(venue_index) {
            match side {
                Side::Buy => *cash -= price * size + fee,
                Side::Sell => *cash += price * size - fee,
            }
        }
        self.touched_accounts.insert(venue_index);
    }

    fn decide_fill(&self, place: &crate::types::PlaceOrderIntent) -> FillDecision {
        if self.config.fill_mode == PaperFillMode::None {
            return FillDecision::none();
        }
        let (best_bid, best_ask, mid) = self.best_prices(place.venue_index);
        let marketable = match place.side {
            Side::Buy => best_ask.map_or(false, |ask| place.price >= ask),
            Side::Sell => best_bid.map_or(false, |bid| place.price <= bid),
        };
        match self.config.fill_mode {
            PaperFillMode::None => FillDecision::none(),
            PaperFillMode::Marketable => {
                if marketable {
                    FillDecision::price(
                        self.price_with_slippage(place.side, best_ask, best_bid),
                        true,
                    )
                } else {
                    FillDecision::none()
                }
            }
            PaperFillMode::Mid => {
                let base = mid.unwrap_or(place.price);
                FillDecision::price(self.price_with_slippage_mid(place.side, base), marketable)
            }
            PaperFillMode::Always => {
                let price = match (best_bid, best_ask, mid) {
                    (_, _, Some(mid)) => self.price_with_slippage_mid(place.side, mid),
                    (Some(bid), Some(ask), None) => {
                        let mid = (bid + ask) * 0.5;
                        self.price_with_slippage_mid(place.side, mid)
                    }
                    (Some(bid), None, None) => self.price_with_slippage_mid(place.side, bid),
                    (None, Some(ask), None) => self.price_with_slippage_mid(place.side, ask),
                    (None, None, None) => place.price,
                };
                FillDecision::price(price, marketable)
            }
        }
    }

    fn decide_fill_for_order(&self, order: &PaperOrder) -> FillDecision {
        if self.config.fill_mode == PaperFillMode::None {
            return FillDecision::none();
        }
        let (best_bid, best_ask, mid) = self.best_prices(order.venue_index);
        let marketable = match order.side {
            Side::Buy => best_ask.map_or(false, |ask| order.price >= ask),
            Side::Sell => best_bid.map_or(false, |bid| order.price <= bid),
        };
        match self.config.fill_mode {
            PaperFillMode::None => FillDecision::none(),
            PaperFillMode::Marketable => {
                if marketable {
                    FillDecision::price(
                        self.price_with_slippage(order.side, best_ask, best_bid),
                        true,
                    )
                } else {
                    FillDecision::none()
                }
            }
            PaperFillMode::Mid => {
                let base = mid.unwrap_or(order.price);
                FillDecision::price(self.price_with_slippage_mid(order.side, base), marketable)
            }
            PaperFillMode::Always => {
                let price = match (best_bid, best_ask, mid) {
                    (_, _, Some(mid)) => self.price_with_slippage_mid(order.side, mid),
                    (Some(bid), Some(ask), None) => {
                        let mid = (bid + ask) * 0.5;
                        self.price_with_slippage_mid(order.side, mid)
                    }
                    (Some(bid), None, None) => self.price_with_slippage_mid(order.side, bid),
                    (None, Some(ask), None) => self.price_with_slippage_mid(order.side, ask),
                    (None, None, None) => order.price,
                };
                FillDecision::price(price, marketable)
            }
        }
    }

    fn best_prices(&self, venue_index: usize) -> (Option<f64>, Option<f64>, Option<f64>) {
        let Some((best_bid, best_ask)) = self.best_bid_ask.get(&venue_index).copied() else {
            return (None, None, None);
        };
        let mid = match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => Some((bid + ask) * 0.5),
            _ => None,
        };
        (best_bid, best_ask, mid)
    }

    fn price_with_slippage(&self, side: Side, best_ask: Option<f64>, best_bid: Option<f64>) -> f64 {
        let base = match side {
            Side::Buy => best_ask.or(best_bid),
            Side::Sell => best_bid.or(best_ask),
        }
        .unwrap_or(0.0);
        self.price_with_slippage_mid(side, base)
    }

    fn price_with_slippage_mid(&self, side: Side, base: f64) -> f64 {
        let slip = self.config.slippage_bps / 10_000.0;
        match side {
            Side::Buy => base * (1.0 + slip),
            Side::Sell => base * (1.0 - slip),
        }
    }

    fn fee_bps(&self, venue_index: usize, marketable: bool) -> f64 {
        let (maker_fee, taker_fee) = self.fees.get(venue_index).copied().unwrap_or((0.0, 0.0));
        if marketable {
            taker_fee
        } else {
            maker_fee
        }
    }

    fn next_order_id(
        &mut self,
        venue_id: &str,
        tick: u64,
        side: Side,
        purpose: OrderPurpose,
        provided: Option<String>,
    ) -> String {
        if let Some(id) = provided {
            return id;
        }
        let nonce = self.nonce;
        self.nonce = self.nonce.wrapping_add(1);
        format!(
            "paper_{:016x}_{}_{}_{}_{}_{}",
            self.cfg_hash,
            venue_id,
            tick,
            format!("{:?}", side).to_lowercase(),
            format!("{:?}", purpose).to_lowercase(),
            nonce
        )
    }

    fn crosses_book(&self, place: &crate::types::PlaceOrderIntent) -> bool {
        let Some((best_bid, best_ask)) = self.best_bid_ask.get(&place.venue_index) else {
            return false;
        };
        match place.side {
            Side::Buy => best_ask.map_or(false, |ask| place.price >= ask),
            Side::Sell => best_bid.map_or(false, |bid| place.price <= bid),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FillDecision {
    fill_price: Option<f64>,
    marketable: bool,
}

impl FillDecision {
    fn none() -> Self {
        Self {
            fill_price: None,
            marketable: false,
        }
    }

    fn price(price: f64, marketable: bool) -> Self {
        Self {
            fill_price: Some(price),
            marketable,
        }
    }
}
