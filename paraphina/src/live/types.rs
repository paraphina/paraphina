//! Live trading event types (feature-gated).

use serde::{Deserialize, Serialize};

use crate::types::{FundingSource, OrderPurpose, SettlementPriceKind, Side, TimestampMs};

use super::orderbook_l2::{BookLevel, BookLevelDelta};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct L2Snapshot {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopOfBook {
    pub best_bid_px: f64,
    pub best_bid_sz: f64,
    pub best_ask_px: f64,
    pub best_ask_sz: f64,
    pub timestamp_ms: Option<TimestampMs>,
}

impl TopOfBook {
    pub fn from_levels(
        bids: &[BookLevel],
        asks: &[BookLevel],
        timestamp_ms: Option<TimestampMs>,
    ) -> Option<Self> {
        let best_bid = bids.iter().find(|lvl| lvl.size > 0.0)?;
        let best_ask = asks.iter().find(|lvl| lvl.size > 0.0)?;
        if !best_bid.price.is_finite()
            || !best_ask.price.is_finite()
            || !best_bid.size.is_finite()
            || !best_ask.size.is_finite()
        {
            return None;
        }
        Some(Self {
            best_bid_px: best_bid.price,
            best_bid_sz: best_bid.size,
            best_ask_px: best_ask.price,
            best_ask_sz: best_ask.size,
            timestamp_ms,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct L2Delta {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub changes: Vec<BookLevelDelta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradeUpdate {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub side: Side,
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FundingUpdate {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub received_ms: Option<TimestampMs>,
    pub funding_rate_8h: Option<f64>,
    pub funding_rate_native: Option<f64>,
    pub interval_sec: Option<u64>,
    pub next_funding_ms: Option<TimestampMs>,
    pub settlement_price_kind: Option<SettlementPriceKind>,
    #[serde(default)]
    pub source: FundingSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MarketDataEvent {
    L2Snapshot(L2Snapshot),
    L2Delta(L2Delta),
    Trade(TradeUpdate),
    FundingUpdate(FundingUpdate),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionSnapshot {
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BalanceSnapshot {
    pub asset: String,
    pub total: f64,
    pub available: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarginSnapshot {
    pub balance_usd: f64,
    pub used_usd: f64,
    pub available_usd: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiquidationSnapshot {
    pub price_liq: Option<f64>,
    pub dist_liq_sigma: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub positions: Vec<PositionSnapshot>,
    pub balances: Vec<BalanceSnapshot>,
    pub funding_8h: Option<f64>,
    pub margin: MarginSnapshot,
    pub liquidation: LiquidationSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccountEvent {
    Snapshot(AccountSnapshot),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderAccepted {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub order_id: String,
    pub client_order_id: Option<String>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderRejected {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub order_id: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
    pub fill_id: Option<String>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    pub fee_bps: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelAccepted {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub order_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelRejected {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub order_id: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelAllAccepted {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelAllRejected {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenOrderSnapshot {
    pub order_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderSnapshot {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub open_orders: Vec<OpenOrderSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionEvent {
    OrderAccepted(OrderAccepted),
    OrderRejected(OrderRejected),
    Filled(Fill),
    CancelAccepted(CancelAccepted),
    CancelRejected(CancelRejected),
    CancelAllAccepted(CancelAllAccepted),
    CancelAllRejected(CancelAllRejected),
    OrderSnapshot(OrderSnapshot),
}
