// src/io/noop.rs
//
// No-op / recording adapter for Paraphina (Milestone H).
//
// Implements VenueAdapter for replay and testing:
// - Records all actions without executing them
// - Useful for determinism tests (record → replay → compare)
// - Can be configured to return specific results for testing

use std::sync::{Arc, Mutex};

use crate::actions::{Action, ActionBatch, CancelAllAction, CancelOrderAction, PlaceOrderAction};
use crate::config::Config;
use crate::io::{ActionResult, VenueAdapter};
use crate::state::GlobalState;

/// Recorded action with metadata.
#[derive(Debug, Clone)]
pub struct RecordedAction {
    /// The action that was executed.
    pub action: Action,
    /// Timestamp when recorded.
    pub timestamp_ms: i64,
    /// Venue index.
    pub venue_index: usize,
}

/// Thread-safe action recorder.
#[derive(Debug, Clone, Default)]
pub struct ActionRecorder {
    actions: Arc<Mutex<Vec<RecordedAction>>>,
}

impl ActionRecorder {
    /// Create a new empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an action.
    pub fn record(&self, action: Action, venue_index: usize, timestamp_ms: i64) {
        if let Ok(mut actions) = self.actions.lock() {
            actions.push(RecordedAction {
                action,
                timestamp_ms,
                venue_index,
            });
        }
    }

    /// Get all recorded actions.
    pub fn get_actions(&self) -> Vec<RecordedAction> {
        self.actions.lock().map(|a| a.clone()).unwrap_or_default()
    }

    /// Get recorded actions as a batch for comparison.
    pub fn as_action_batch(
        &self,
        now_ms: i64,
        tick_index: u64,
        config_version: &str,
    ) -> ActionBatch {
        let mut batch = ActionBatch::new(now_ms, tick_index, config_version);
        for recorded in self.get_actions() {
            batch.push(recorded.action);
        }
        batch
    }

    /// Clear all recorded actions.
    pub fn clear(&self) {
        if let Ok(mut actions) = self.actions.lock() {
            actions.clear();
        }
    }

    /// Get the number of recorded actions.
    pub fn len(&self) -> usize {
        self.actions.lock().map(|a| a.len()).unwrap_or(0)
    }

    /// Check if the recorder is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// No-op adapter that records actions but doesn't execute them.
///
/// Useful for:
/// - Replay determinism tests (compare recorded actions)
/// - Unit testing strategy logic
/// - Dry-run mode
#[derive(Debug, Clone)]
pub struct NoopAdapter {
    /// Venue index this adapter handles.
    venue_index: usize,
    /// Shared recorder for all actions.
    recorder: ActionRecorder,
    /// Current timestamp (for recording).
    current_ms: i64,
}

impl NoopAdapter {
    /// Create a new no-op adapter with a shared recorder.
    pub fn new(venue_index: usize, recorder: ActionRecorder) -> Self {
        Self {
            venue_index,
            recorder,
            current_ms: 0,
        }
    }

    /// Get the shared recorder.
    pub fn recorder(&self) -> &ActionRecorder {
        &self.recorder
    }
}

impl VenueAdapter for NoopAdapter {
    fn set_timestamp(&mut self, now_ms: i64) {
        self.current_ms = now_ms;
    }

    fn place_order(
        &mut self,
        _cfg: &Config,
        _state: &mut GlobalState,
        action: &PlaceOrderAction,
    ) -> ActionResult {
        // Record the action
        self.recorder.record(
            Action::PlaceOrder(action.clone()),
            self.venue_index,
            self.current_ms,
        );

        // Return acknowledged (no actual fill)
        ActionResult::Placed {
            order_id: action.client_order_id.clone(),
        }
    }

    fn cancel_order(
        &mut self,
        _cfg: &Config,
        _state: &mut GlobalState,
        action: &CancelOrderAction,
    ) -> ActionResult {
        self.recorder.record(
            Action::CancelOrder(action.clone()),
            self.venue_index,
            self.current_ms,
        );

        ActionResult::Cancelled {
            order_id: action.order_id.clone(),
        }
    }

    fn cancel_all(
        &mut self,
        _cfg: &Config,
        _state: &mut GlobalState,
        action: &CancelAllAction,
    ) -> ActionResult {
        self.recorder.record(
            Action::CancelAll(action.clone()),
            self.venue_index,
            self.current_ms,
        );

        ActionResult::CancelledAll { count: 0 }
    }

    fn name(&self) -> &str {
        "NoopAdapter"
    }
}

/// Create no-op adapters for all venues with a shared recorder.
pub fn create_noop_adapters(cfg: &Config) -> (Vec<Box<dyn VenueAdapter>>, ActionRecorder) {
    let recorder = ActionRecorder::new();
    let adapters: Vec<Box<dyn VenueAdapter>> = cfg
        .venues
        .iter()
        .enumerate()
        .map(|(i, _)| Box::new(NoopAdapter::new(i, recorder.clone())) as Box<dyn VenueAdapter>)
        .collect();

    (adapters, recorder)
}

/// Compare two action batches for equality.
///
/// Returns Ok(()) if equal, or Err with a description of differences.
pub fn compare_batches(a: &ActionBatch, b: &ActionBatch) -> Result<(), String> {
    if a.actions.len() != b.actions.len() {
        return Err(format!(
            "Action count mismatch: {} vs {}",
            a.actions.len(),
            b.actions.len()
        ));
    }

    for (i, (action_a, action_b)) in a.actions.iter().zip(b.actions.iter()).enumerate() {
        if action_a != action_b {
            return Err(format!(
                "Action {} differs:\n  A: {:?}\n  B: {:?}",
                i, action_a, action_b
            ));
        }
    }

    Ok(())
}

/// Compare recorded actions from two recorders.
pub fn compare_recordings(
    recorder_a: &ActionRecorder,
    recorder_b: &ActionRecorder,
) -> Result<(), String> {
    let actions_a = recorder_a.get_actions();
    let actions_b = recorder_b.get_actions();

    if actions_a.len() != actions_b.len() {
        return Err(format!(
            "Recording count mismatch: {} vs {}",
            actions_a.len(),
            actions_b.len()
        ));
    }

    for (i, (ra, rb)) in actions_a.iter().zip(actions_b.iter()).enumerate() {
        if ra.action != rb.action {
            return Err(format!(
                "Recorded action {} differs:\n  A: {:?}\n  B: {:?}",
                i, ra.action, rb.action
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OrderPurpose, Side};

    #[test]
    fn test_action_recorder() {
        let recorder = ActionRecorder::new();
        assert!(recorder.is_empty());

        let action = Action::PlaceOrder(PlaceOrderAction {
            action_id: "test_001".to_string(),
            venue_index: 0,
            venue_id: "test".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            client_order_id: "co_test_001".to_string(),
        });

        recorder.record(action.clone(), 0, 1000);

        assert_eq!(recorder.len(), 1);
        assert!(!recorder.is_empty());

        let actions = recorder.get_actions();
        assert_eq!(actions[0].action, action);
        assert_eq!(actions[0].venue_index, 0);
        assert_eq!(actions[0].timestamp_ms, 1000);
    }

    #[test]
    fn test_noop_adapter_records() {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        let recorder = ActionRecorder::new();
        let mut adapter = NoopAdapter::new(0, recorder.clone());

        let action = PlaceOrderAction {
            action_id: "test_002".to_string(),
            venue_index: 0,
            venue_id: "extended".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            client_order_id: "co_test_002".to_string(),
        };

        adapter.set_timestamp(2000);
        let result = adapter.place_order(&cfg, &mut state, &action);

        // Should return Placed, not Filled
        match result {
            ActionResult::Placed { order_id } => {
                assert_eq!(order_id, "co_test_002");
            }
            _ => panic!("Expected Placed result"),
        }

        // Should be recorded
        assert_eq!(recorder.len(), 1);

        // State should NOT be modified (no-op)
        assert_eq!(state.venues[0].position_tao, 0.0);
    }

    #[test]
    fn test_compare_batches_equal() {
        let mut batch_a = ActionBatch::new(1000, 0, "v0.1.0");
        let mut batch_b = ActionBatch::new(1000, 0, "v0.1.0");

        let action = Action::PlaceOrder(PlaceOrderAction {
            action_id: "test_003".to_string(),
            venue_index: 0,
            venue_id: "test".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            client_order_id: "co_test_003".to_string(),
        });

        batch_a.push(action.clone());
        batch_b.push(action);

        assert!(compare_batches(&batch_a, &batch_b).is_ok());
    }

    #[test]
    fn test_compare_batches_different_count() {
        let mut batch_a = ActionBatch::new(1000, 0, "v0.1.0");
        let batch_b = ActionBatch::new(1000, 0, "v0.1.0");

        batch_a.push(Action::PlaceOrder(PlaceOrderAction {
            action_id: "test_004".to_string(),
            venue_index: 0,
            venue_id: "test".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            client_order_id: "co_test_004".to_string(),
        }));

        let result = compare_batches(&batch_a, &batch_b);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("count mismatch"));
    }

    #[test]
    fn test_compare_batches_different_action() {
        let mut batch_a = ActionBatch::new(1000, 0, "v0.1.0");
        let mut batch_b = ActionBatch::new(1000, 0, "v0.1.0");

        batch_a.push(Action::PlaceOrder(PlaceOrderAction {
            action_id: "test_005".to_string(),
            venue_index: 0,
            venue_id: "test".to_string(),
            side: Side::Buy,
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            client_order_id: "co_test_005".to_string(),
        }));

        batch_b.push(Action::PlaceOrder(PlaceOrderAction {
            action_id: "test_005".to_string(),
            venue_index: 0,
            venue_id: "test".to_string(),
            side: Side::Sell, // Different side
            price: 100.0,
            size: 1.0,
            purpose: OrderPurpose::Mm,
            client_order_id: "co_test_005".to_string(),
        }));

        let result = compare_batches(&batch_a, &batch_b);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("differs"));
    }

    #[test]
    fn test_create_noop_adapters() {
        let cfg = Config::default();
        let (adapters, recorder) = create_noop_adapters(&cfg);

        assert_eq!(adapters.len(), cfg.venues.len());
        assert!(recorder.is_empty());

        for adapter in &adapters {
            assert_eq!(adapter.name(), "NoopAdapter");
        }
    }
}
