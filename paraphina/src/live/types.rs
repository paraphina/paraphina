//! Live trading event types (feature-gated).

use std::fmt;

use serde::{Deserialize, Serialize};

pub use crate::types::Phase51ForwardRefreshTargetKey;
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
    #[serde(default)]
    pub order_id: Option<String>,
    #[serde(default)]
    pub client_order_id: Option<String>,
    #[serde(default)]
    pub purpose: Option<OrderPurpose>,
    #[serde(default)]
    pub reduce_only: Option<bool>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase51ForwardRefreshNativeRole {
    Aster {
        maker: bool,
        last_filled_qty: String,
    },
    Extended {
        is_taker: bool,
    },
    Hyperliquid {
        crossed: bool,
    },
    Lighter {
        account_index: i64,
        is_maker_ask: bool,
        ask_account_id: i64,
        bid_account_id: i64,
    },
    Paradex {
        liquidity: String,
    },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase51ForwardRefreshSourceOwnerFill {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    #[serde(default, skip_serializing, skip_deserializing)]
    order_id: Option<String>,
    #[serde(default, skip_serializing, skip_deserializing)]
    client_order_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_target_key: Option<Phase51ForwardRefreshTargetKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_native_role: Option<Phase51ForwardRefreshNativeRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_lighter_native_limit: Option<Phase51ForwardRefreshLighterNativeLimit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_source_owner_pfill_observation:
        Option<Phase51ForwardRefreshSourceOwnerPfillObservation>,
}

impl Phase51ForwardRefreshSourceOwnerFill {
    pub fn new(
        venue_index: usize,
        venue_id: impl Into<String>,
        seq: u64,
        timestamp_ms: TimestampMs,
        order_id: Option<String>,
        client_order_id: Option<String>,
        phase51_native_role: Option<Phase51ForwardRefreshNativeRole>,
    ) -> Self {
        Self {
            venue_index,
            venue_id: venue_id.into(),
            seq,
            timestamp_ms,
            order_id,
            client_order_id,
            phase51_target_key: None,
            phase51_native_role,
            phase51_lighter_native_limit: None,
            phase51_source_owner_pfill_observation: None,
        }
    }

    pub fn order_id(&self) -> Option<&str> {
        self.order_id.as_deref()
    }

    pub fn client_order_id(&self) -> Option<&str> {
        self.client_order_id.as_deref()
    }

    pub fn set_phase51_target_key(&mut self, target_key: Phase51ForwardRefreshTargetKey) {
        self.phase51_target_key = Some(target_key);
    }

    pub fn set_phase51_source_owner_pfill_observation(
        &mut self,
        observation: Phase51ForwardRefreshSourceOwnerPfillObservation,
    ) {
        self.phase51_source_owner_pfill_observation = Some(observation);
    }
}

impl fmt::Debug for Phase51ForwardRefreshSourceOwnerFill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Phase51ForwardRefreshSourceOwnerFill")
            .field("venue_index", &self.venue_index)
            .field("venue_id", &self.venue_id)
            .field("seq", &self.seq)
            .field("timestamp_ms", &self.timestamp_ms)
            .field("order_id", &self.order_id.as_ref().map(|_| "<redacted>"))
            .field(
                "client_order_id",
                &self.client_order_id.as_ref().map(|_| "<redacted>"),
            )
            .field("phase51_target_key", &self.phase51_target_key)
            .field("phase51_native_role", &self.phase51_native_role)
            .field(
                "phase51_lighter_native_limit",
                &self.phase51_lighter_native_limit,
            )
            .field(
                "phase51_source_owner_pfill_observation",
                &self.phase51_source_owner_pfill_observation,
            )
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase51ForwardRefreshSourceOwnerPfillObservation {
    pub source_event_type: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub event_time_ms: TimestampMs,
    pub fill_count: u32,
    pub outcome_status: String,
    pub p_fill_outcome: f64,
}

impl Phase51ForwardRefreshSourceOwnerPfillObservation {
    pub fn lighter_trade_observed_fill(
        side: Side,
        price: f64,
        size: f64,
        event_time_ms: TimestampMs,
    ) -> Option<Self> {
        if !price.is_finite() || price <= 0.0 {
            return None;
        }
        if !size.is_finite() || size <= 0.0 {
            return None;
        }
        if event_time_ms <= 0 {
            return None;
        }
        Some(Self {
            source_event_type: "LIGHTER_TRADES_JSON".to_string(),
            side,
            price,
            size,
            event_time_ms,
            fill_count: 1,
            outcome_status: "OBSERVED_FILLED".to_string(),
            p_fill_outcome: 1.0,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Phase51ForwardRefreshLighterNativeLimit {
    pub active_order_headroom_account: Option<i64>,
    pub active_order_sendtx_utilization_account: Option<i64>,
    pub rest_open_orders_count: Option<i64>,
    pub rest_open_orders_cap: Option<i64>,
    pub weighted_open_order_slots_used: Option<i64>,
    pub weighted_open_order_slots_cap: Option<i64>,
    pub native_limit_event_time_status: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_target_key: Option<Phase51ForwardRefreshTargetKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_native_role: Option<Phase51ForwardRefreshNativeRole>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase51_lighter_native_limit: Option<Phase51ForwardRefreshLighterNativeLimit>,
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
    pub client_order_id: Option<String>,
    pub exchange_order_id: Option<String>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: Option<OrderPurpose>,
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
    Phase51ForwardRefreshSourceOwnerFill(Phase51ForwardRefreshSourceOwnerFill),
    CancelAccepted(CancelAccepted),
    CancelRejected(CancelRejected),
    CancelAllAccepted(CancelAllAccepted),
    CancelAllRejected(CancelAllRejected),
    OrderSnapshot(OrderSnapshot),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase51ForwardRefreshCaptureAudit {
    pub enabled: bool,
    pub target_type: Option<String>,
    pub venue_id: Option<String>,
    pub canonical_group_id: Option<String>,
    pub order_key: Option<String>,
    pub native_role_source: Option<String>,
    pub lighter_pressure_status: Option<String>,
    pub sanitized_row_emitted: bool,
    pub no_live_flag: bool,
    pub approved_for_live: bool,
    pub approved_for_canary: bool,
    pub approved_for_capital_escalation: bool,
}
