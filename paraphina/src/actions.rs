// src/actions.rs
//
// Core Action types for Paraphina (Milestone H).
//
// These types represent the deterministic, pure output of the strategy core.
// The strategy emits Actions; the I/O layer (VenueAdapter/Gateway) executes them.
//
// Design principles:
// - All Actions are deterministic given the same state + config + timestamp.
// - Action IDs are computed deterministically from tick_index, venue_index, and purpose.
// - Actions are serializable for replay testing and debugging.
// - Actions derive Debug + Clone + PartialEq for test assertions.

use serde::{Deserialize, Serialize};

use crate::state::KillReason;
use crate::types::{OrderIntent, OrderPurpose, Side, TimeInForce, TimestampMs};

/// Unique identifier for an action, computed deterministically.
///
/// Format: `{tick_index}_{venue_index}_{purpose}_{sequence}`
/// - tick_index: monotonic tick counter
/// - venue_index: target venue (0 for global actions)
/// - purpose: order purpose code (mm/exit/hedge/system)
/// - sequence: per-tick sequence number for disambiguation
pub type ActionId = String;

/// Core action types emitted by the strategy.
///
/// The strategy core computes what *should* happen; the gateway executes it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Place a new order on a venue.
    PlaceOrder(PlaceOrderAction),

    /// Cancel an existing order.
    CancelOrder(CancelOrderAction),

    /// Cancel all orders, optionally scoped to a venue.
    CancelAll(CancelAllAction),

    /// Transition risk state / activate kill switch.
    SetKillSwitch(SetKillSwitchAction),

    /// Log a message (for deterministic replay and debugging).
    Log(LogAction),
}

/// Place a new order on a venue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaceOrderAction {
    /// Deterministic action ID.
    pub action_id: ActionId,
    /// Target venue index.
    pub venue_index: usize,
    /// Stable venue identifier.
    pub venue_id: String,
    /// Buy or sell.
    pub side: Side,
    /// Limit price.
    pub price: f64,
    /// Size in TAO.
    pub size: f64,
    /// Order purpose (MM, Exit, Hedge).
    pub purpose: OrderPurpose,
    /// Time-in-force policy (GTC/IOC).
    pub time_in_force: crate::types::TimeInForce,
    /// Post-only flag (reject if crosses).
    pub post_only: bool,
    /// Reduce-only flag (never increase position).
    pub reduce_only: bool,
    /// Client order ID for tracking (deterministic).
    pub client_order_id: String,
}

/// Cancel a specific order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelOrderAction {
    /// Deterministic action ID.
    pub action_id: ActionId,
    /// Target venue index.
    pub venue_index: usize,
    /// Stable venue identifier.
    pub venue_id: String,
    /// Order ID or client order ID to cancel.
    pub order_id: String,
}

/// Cancel all orders, optionally scoped to a venue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CancelAllAction {
    /// Deterministic action ID.
    pub action_id: ActionId,
    /// If Some, cancel only on this venue. If None, cancel globally.
    pub venue_index: Option<usize>,
}

/// Activate or transition the kill switch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetKillSwitchAction {
    /// Deterministic action ID.
    pub action_id: ActionId,
    /// Whether to activate (true) or deactivate (false) the kill switch.
    pub activate: bool,
    /// Reason for kill switch activation.
    pub reason: KillReason,
}

/// Log level for Log actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Log a message (deterministic, for replay).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogAction {
    /// Deterministic action ID.
    pub action_id: ActionId,
    /// Log level.
    pub level: LogLevel,
    /// Log message.
    pub message: String,
}

/// A batch of actions with metadata for replay and determinism verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionBatch {
    /// Timestamp when this batch was computed.
    pub now_ms: TimestampMs,
    /// Tick index (monotonic counter).
    pub tick_index: u64,
    /// Config version identifier (for drift detection).
    pub config_version: String,
    /// Optional deterministic seed used for this run.
    pub run_seed: Option<u64>,
    /// The actions in this batch.
    pub actions: Vec<Action>,
}

impl ActionBatch {
    /// Create a new empty action batch.
    pub fn new(now_ms: TimestampMs, tick_index: u64, config_version: &str) -> Self {
        Self {
            now_ms,
            tick_index,
            config_version: config_version.to_string(),
            run_seed: None,
            actions: Vec::new(),
        }
    }

    /// Set the run seed.
    pub fn with_seed(mut self, seed: Option<u64>) -> Self {
        self.run_seed = seed;
        self
    }

    /// Add an action to the batch.
    pub fn push(&mut self, action: Action) {
        self.actions.push(action);
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Get the number of actions.
    pub fn len(&self) -> usize {
        self.actions.len()
    }
}

/// Context for generating deterministic action IDs.
#[derive(Debug, Clone)]
pub struct ActionIdGenerator {
    tick_index: u64,
    sequence: u32,
}

impl ActionIdGenerator {
    /// Create a new ID generator for a tick.
    pub fn new(tick_index: u64) -> Self {
        Self {
            tick_index,
            sequence: 0,
        }
    }

    /// Generate a deterministic action ID.
    ///
    /// Format: `t{tick}_{purpose}_{venue}_{seq}`
    pub fn next_id(&mut self, venue_index: usize, purpose: &str) -> ActionId {
        let id = format!(
            "t{}_{}_v{}_{:04}",
            self.tick_index, purpose, venue_index, self.sequence
        );
        self.sequence += 1;
        id
    }

    /// Generate a client order ID for tracking.
    ///
    /// Format: `co_{tick}_{venue}_{purpose}_{seq}`
    pub fn client_order_id(&mut self, venue_index: usize, purpose: OrderPurpose) -> String {
        let purpose_str = match purpose {
            OrderPurpose::Mm => "mm",
            OrderPurpose::Exit => "exit",
            OrderPurpose::Hedge => "hedge",
        };
        let id = format!(
            "co_{}_v{}_{}_{}",
            self.tick_index, venue_index, purpose_str, self.sequence
        );
        self.sequence += 1;
        id
    }
}

/// Convert order intents into deterministic action list.
pub fn intents_to_actions(intents: &[OrderIntent], gen: &mut ActionIdGenerator) -> Vec<Action> {
    let mut actions = Vec::new();
    for intent in intents {
        match intent {
            OrderIntent::Place(place) => {
                let purpose_str = match place.purpose {
                    OrderPurpose::Mm => "mm",
                    OrderPurpose::Exit => "exit",
                    OrderPurpose::Hedge => "hedge",
                };
                let action_id = gen.next_id(place.venue_index, purpose_str);
                let client_order_id = place
                    .client_order_id
                    .clone()
                    .unwrap_or_else(|| gen.client_order_id(place.venue_index, place.purpose));
                actions.push(Action::PlaceOrder(PlaceOrderAction {
                    action_id,
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
                }));
            }
            OrderIntent::Cancel(cancel) => {
                let action_id = gen.next_id(cancel.venue_index, "cancel");
                actions.push(Action::CancelOrder(CancelOrderAction {
                    action_id,
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.to_string(),
                    order_id: cancel.order_id.clone(),
                }));
            }
            OrderIntent::Replace(replace) => {
                let cancel_action_id = gen.next_id(replace.venue_index, "cancel");
                actions.push(Action::CancelOrder(CancelOrderAction {
                    action_id: cancel_action_id,
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.to_string(),
                    order_id: replace.order_id.clone(),
                }));
                let purpose_str = match replace.purpose {
                    OrderPurpose::Mm => "mm",
                    OrderPurpose::Exit => "exit",
                    OrderPurpose::Hedge => "hedge",
                };
                let place_action_id = gen.next_id(replace.venue_index, purpose_str);
                let client_order_id = replace
                    .client_order_id
                    .clone()
                    .unwrap_or_else(|| gen.client_order_id(replace.venue_index, replace.purpose));
                actions.push(Action::PlaceOrder(PlaceOrderAction {
                    action_id: place_action_id,
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.to_string(),
                    side: replace.side,
                    price: replace.price,
                    size: replace.size,
                    purpose: replace.purpose,
                    time_in_force: replace.time_in_force,
                    post_only: replace.post_only,
                    reduce_only: replace.reduce_only,
                    client_order_id,
                }));
            }
            OrderIntent::CancelAll(cancel_all) => {
                let action_id = gen.next_id(cancel_all.venue_index.unwrap_or(0), "cancel_all");
                actions.push(Action::CancelAll(crate::actions::CancelAllAction {
                    action_id,
                    venue_index: cancel_all.venue_index,
                }));
            }
        }
    }
    actions
}

/// Builder for creating actions with deterministic IDs.
pub struct ActionBuilder<'a> {
    gen: &'a mut ActionIdGenerator,
}

impl<'a> ActionBuilder<'a> {
    /// Create a new action builder.
    pub fn new(gen: &'a mut ActionIdGenerator) -> Self {
        Self { gen }
    }

    /// Build a PlaceOrder action.
    pub fn place_order(
        &mut self,
        venue_index: usize,
        venue_id: &str,
        side: Side,
        price: f64,
        size: f64,
        purpose: OrderPurpose,
    ) -> Action {
        let purpose_str = match purpose {
            OrderPurpose::Mm => "mm",
            OrderPurpose::Exit => "exit",
            OrderPurpose::Hedge => "hedge",
        };
        let action_id = self.gen.next_id(venue_index, purpose_str);
        let client_order_id = self.gen.client_order_id(venue_index, purpose);

        Action::PlaceOrder(PlaceOrderAction {
            action_id,
            venue_index,
            venue_id: venue_id.to_string(),
            side,
            price,
            size,
            purpose,
            time_in_force: TimeInForce::Gtc,
            post_only: false,
            reduce_only: false,
            client_order_id,
        })
    }

    /// Build a CancelOrder action.
    pub fn cancel_order(&mut self, venue_index: usize, venue_id: &str, order_id: &str) -> Action {
        let action_id = self.gen.next_id(venue_index, "cancel");
        Action::CancelOrder(CancelOrderAction {
            action_id,
            venue_index,
            venue_id: venue_id.to_string(),
            order_id: order_id.to_string(),
        })
    }

    /// Build a CancelAll action.
    pub fn cancel_all(&mut self, venue_index: Option<usize>) -> Action {
        let v_idx = venue_index.unwrap_or(0);
        let action_id = self.gen.next_id(v_idx, "cancel_all");
        Action::CancelAll(CancelAllAction {
            action_id,
            venue_index,
        })
    }

    /// Build a SetKillSwitch action.
    pub fn set_kill_switch(&mut self, activate: bool, reason: KillReason) -> Action {
        let action_id = self.gen.next_id(0, "kill_switch");
        Action::SetKillSwitch(SetKillSwitchAction {
            action_id,
            activate,
            reason,
        })
    }

    /// Build a Log action.
    pub fn log(&mut self, level: LogLevel, message: &str) -> Action {
        let action_id = self.gen.next_id(0, "log");
        Action::Log(LogAction {
            action_id,
            level,
            message: message.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_id_generator_deterministic() {
        let mut gen1 = ActionIdGenerator::new(42);
        let mut gen2 = ActionIdGenerator::new(42);

        // Same inputs should produce same outputs
        let id1_a = gen1.next_id(0, "mm");
        let id1_b = gen1.next_id(1, "hedge");

        let id2_a = gen2.next_id(0, "mm");
        let id2_b = gen2.next_id(1, "hedge");

        assert_eq!(id1_a, id2_a);
        assert_eq!(id1_b, id2_b);
    }

    #[test]
    fn test_action_id_format() {
        let mut gen = ActionIdGenerator::new(100);
        let id = gen.next_id(2, "mm");
        assert_eq!(id, "t100_mm_v2_0000");

        let id2 = gen.next_id(3, "hedge");
        assert_eq!(id2, "t100_hedge_v3_0001");
    }

    #[test]
    fn test_action_builder() {
        let mut gen = ActionIdGenerator::new(5);
        let mut builder = ActionBuilder::new(&mut gen);

        let action = builder.place_order(0, "extended", Side::Buy, 100.0, 1.0, OrderPurpose::Mm);

        match action {
            Action::PlaceOrder(po) => {
                assert_eq!(po.venue_index, 0);
                assert_eq!(po.venue_id, "extended");
                assert_eq!(po.side, Side::Buy);
                assert_eq!(po.price, 100.0);
                assert_eq!(po.size, 1.0);
                assert_eq!(po.purpose, OrderPurpose::Mm);
                assert!(po.action_id.starts_with("t5_mm_v0_"));
            }
            _ => panic!("Expected PlaceOrder"),
        }
    }

    #[test]
    fn test_action_batch() {
        let mut batch = ActionBatch::new(1000, 0, "v0.1.0");
        assert!(batch.is_empty());

        let mut gen = ActionIdGenerator::new(0);
        let mut builder = ActionBuilder::new(&mut gen);

        batch.push(builder.place_order(0, "test", Side::Buy, 100.0, 1.0, OrderPurpose::Mm));
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_action_serialization() {
        let mut gen = ActionIdGenerator::new(1);
        let mut builder = ActionBuilder::new(&mut gen);
        let action = builder.place_order(0, "test", Side::Sell, 150.0, 2.5, OrderPurpose::Exit);

        // Should serialize without error
        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("PlaceOrder"));
        assert!(json.contains("150.0"));

        // Should deserialize back
        let parsed: Action = serde_json::from_str(&json).unwrap();
        assert_eq!(action, parsed);
    }
}
