// src/types.rs
//
// Common shared types for the Paraphina MM engine.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Millisecond timestamp since Unix epoch.
pub type TimestampMs = i64;

/// Health status of a venue used by the strategy & risk engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VenueStatus {
    Healthy,
    Warning,  // used for "medium" toxicity / soft risk clamp
    Disabled, // venue is turned off
}

/// Buy or sell side for an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// High-level reason for an order.
/// - Mm    = passive market-making quote
/// - Exit  = cross-venue exit / arb
/// - Hedge = global hedge adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderPurpose {
    Mm,
    Exit,
    Hedge,
}

/// Time-in-force policy for an order intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Immediate-or-cancel (IOC) intent.
    Ioc,
    /// Good-til-cancel (GTC) intent (resting).
    Gtc,
}

/// Abstract order intent: "we want to do X on venue Y".
/// The execution / gateway layer will later turn this into real API calls.
///
/// Note: `venue_id` uses `Arc<str>` for cheap cloning in hot paths.
/// The Arc points to the same string as `VenueConfig.id_arc`.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderIntent {
    Place(PlaceOrderIntent),
    Cancel(CancelOrderIntent),
    Replace(ReplaceOrderIntent),
    CancelAll(CancelAllOrderIntent),
}

/// Place a new order.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaceOrderIntent {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    pub time_in_force: TimeInForce,
    pub post_only: bool,
    pub reduce_only: bool,
    /// Optional deterministic client order ID (used for MM tracking).
    pub client_order_id: Option<String>,
}

/// Cancel an existing order.
#[derive(Debug, Clone, PartialEq)]
pub struct CancelOrderIntent {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub order_id: String,
}

/// Cancel all orders (optionally scoped to a venue).
#[derive(Debug, Clone, PartialEq)]
pub struct CancelAllOrderIntent {
    pub venue_index: Option<usize>,
    pub venue_id: Option<Arc<str>>,
}

/// Replace an existing order (cancel + place).
#[derive(Debug, Clone, PartialEq)]
pub struct ReplaceOrderIntent {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    pub time_in_force: TimeInForce,
    pub post_only: bool,
    pub reduce_only: bool,
    /// Order ID to replace.
    pub order_id: String,
    /// Optional deterministic client order ID for the new order.
    pub client_order_id: Option<String>,
}

/// Execution events emitted by the gateway.
#[derive(Debug, Clone)]
pub enum ExecutionEvent {
    BookUpdate(BookUpdate),
    Fill(FillEvent),
    OrderAck(OrderAck),
    OrderReject(OrderReject),
    BalanceUpdate(BalanceUpdate),
    FundingUpdate(FundingUpdate),
}

#[derive(Debug, Clone)]
pub struct BookUpdate {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub mid: f64,
    pub spread: f64,
    pub depth_near_mid: f64,
    pub timestamp_ms: TimestampMs,
}

#[derive(Debug, Clone)]
pub struct OrderAck {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub order_id: String,
    pub client_order_id: Option<String>,
    pub seq: Option<u64>,
    pub side: Option<Side>,
    pub price: Option<f64>,
    pub size: Option<f64>,
    pub purpose: Option<OrderPurpose>,
}

#[derive(Debug, Clone)]
pub struct OrderReject {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
    pub seq: Option<u64>,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct BalanceUpdate {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub margin_balance_usd: f64,
    pub margin_used_usd: f64,
    pub margin_available_usd: f64,
}

#[derive(Debug, Clone)]
pub struct FundingUpdate {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub funding_8h: f64,
}

/// A realised perp fill used for logging and PnL attribution.
///
/// Note: `venue_id` uses `Arc<str>` for cheap cloning in hot paths.
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
    pub seq: Option<u64>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    /// Net fee in basis points (positive = cost, negative = rebate).
    pub fee_bps: f64,
}
