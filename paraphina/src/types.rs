// src/types.rs
//
// Common shared types for the Paraphina MM engine.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Millisecond timestamp since Unix epoch.
pub type TimestampMs = i64;

/// Current time in milliseconds since Unix epoch.
pub fn now_ms() -> TimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as TimestampMs
}

/// Health status of a venue used by the strategy & risk engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VenueStatus {
    Healthy,
    Warning,  // used for "medium" toxicity / soft risk clamp
    Disabled, // venue is turned off
}

/// Funding data source provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FundingSource {
    MarketDataWs,
    MarketDataRest,
    AccountSnapshot,
    Derived,
    #[default]
    Unknown,
}

/// Health status of funding data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FundingStatus {
    Healthy,
    Stale,
    Unknown,
}

/// Settlement price basis used for funding computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SettlementPriceKind {
    Oracle,
    Mark,
    Index,
    UsdcOracleAdjusted,
    #[default]
    Unknown,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Phase51ForwardRefreshTargetKey {
    pub canonical_group_id: String,
    pub order_key: String,
}

/// First-class upstream target identity for canonical order labels.
///
/// This object is architecture state, not source-owner evidence. It must be
/// created before order intent construction from explicit in-memory target
/// selection state, and must never be derived from order/client IDs, price,
/// side, size, timing, purpose, generated names, hashes, snapshots, config, or
/// proximity.
#[derive(Clone, PartialEq, Eq)]
pub struct CanonicalTargetIdentity {
    canonical_group_id: String,
    order_key: String,
}

impl CanonicalTargetIdentity {
    pub fn from_explicit(
        canonical_group_id: impl Into<String>,
        order_key: impl Into<String>,
    ) -> Option<Self> {
        let canonical_group_id = canonical_group_id.into();
        let order_key = order_key.into();
        if canonical_group_id.trim().is_empty() || order_key.trim().is_empty() {
            return None;
        }
        Some(Self {
            canonical_group_id,
            order_key,
        })
    }

    pub fn canonical_group_id(&self) -> &str {
        &self.canonical_group_id
    }

    pub fn order_key(&self) -> &str {
        &self.order_key
    }

    pub fn to_phase51_target_key(&self) -> Phase51ForwardRefreshTargetKey {
        Phase51ForwardRefreshTargetKey {
            canonical_group_id: self.canonical_group_id.clone(),
            order_key: self.order_key.clone(),
        }
    }
}

impl std::fmt::Debug for CanonicalTargetIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalTargetIdentity")
            .field("canonical_group_id", &"<redacted>")
            .field("order_key", &"<redacted>")
            .finish()
    }
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
    /// Optional explicit Phase 5.1 target key. This must be carried from
    /// upstream target selection and must never be inferred from order fields.
    pub phase51_target_key: Option<Phase51ForwardRefreshTargetKey>,
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
    /// Optional explicit Phase 5.1 target key. This must be carried from
    /// upstream target selection and must never be inferred from order fields.
    pub phase51_target_key: Option<Phase51ForwardRefreshTargetKey>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_target_identity_requires_complete_fields() {
        assert!(CanonicalTargetIdentity::from_explicit("", "order").is_none());
        assert!(CanonicalTargetIdentity::from_explicit("group", "").is_none());
        assert!(CanonicalTargetIdentity::from_explicit("   ", "order").is_none());
        assert!(CanonicalTargetIdentity::from_explicit("group", "   ").is_none());
        assert!(CanonicalTargetIdentity::from_explicit("group", "order").is_some());
    }

    #[test]
    fn canonical_target_identity_debug_redacts_values() {
        let identity = CanonicalTargetIdentity::from_explicit("sensitive-group", "sensitive-order")
            .expect("complete identity");
        let debug = format!("{identity:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("sensitive-group"));
        assert!(!debug.contains("sensitive-order"));
    }

    #[test]
    fn canonical_target_identity_converts_to_phase51_key() {
        let identity = CanonicalTargetIdentity::from_explicit("canonical-group", "canonical-order")
            .expect("complete identity");
        let target_key = identity.to_phase51_target_key();
        assert_eq!(target_key.canonical_group_id, identity.canonical_group_id());
        assert_eq!(target_key.order_key, identity.order_key());
    }
}

#[derive(Debug, Clone)]
pub struct OrderReject {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub order_id: Option<String>,
    pub client_order_id: Option<String>,
    pub seq: Option<u64>,
    pub purpose: Option<OrderPurpose>,
    pub reduce_only: Option<bool>,
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
