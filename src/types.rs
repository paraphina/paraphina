// src/types.rs
//
// Common shared types for the Paraphina MM engine.

/// Millisecond timestamp since Unix epoch.
pub type TimestampMs = i64;

/// Health status of a venue used by the strategy & risk engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VenueStatus {
    Healthy,
    Warning,  // used for "medium" toxicity / soft risk clamp
    Disabled, // venue is turned off
}

/// Buy or sell side for an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// High-level reason for an order.
/// - Mm    = passive market-making quote
/// - Exit  = cross-venue exit / arb
/// - Hedge = global hedge adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderPurpose {
    Mm,
    Exit,
    Hedge,
}

/// Abstract order intent: "we want to do X on venue Y".
/// The execution / gateway layer will later turn this into real API calls.
#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
}

/// A realised perp fill used for logging and PnL attribution.
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub venue_index: usize,
    pub venue_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub purpose: OrderPurpose,
    /// Net fee in basis points (positive = cost, negative = rebate).
    pub fee_bps: f64,
}
