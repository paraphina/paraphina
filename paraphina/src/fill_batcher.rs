// src/fill_batcher.rs
//
// Milestone H: deterministic fill aggregation for §4.3.

use crate::types::FillEvent;

#[derive(Debug, Clone)]
pub struct FillBatcher {
    interval_ms: i64,
    last_flush_ms: i64,
    pending: Vec<FillEvent>,
}

impl FillBatcher {
    /// Create a new batcher with the given interval.
    pub fn new(interval_ms: i64) -> Self {
        Self {
            interval_ms: interval_ms.max(1),
            last_flush_ms: 0,
            pending: Vec::new(),
        }
    }

    /// Update the last flush timestamp (used to align to tick boundaries).
    pub fn set_last_flush_ms(&mut self, last_flush_ms: i64) {
        self.last_flush_ms = last_flush_ms;
    }

    /// Record fills for the current tick.
    pub fn push(&mut self, _now_ms: i64, fills: Vec<FillEvent>) {
        self.pending.extend(fills);
    }

    /// Return true if the batch window has elapsed.
    pub fn should_flush(&self, now_ms: i64) -> bool {
        now_ms - self.last_flush_ms >= self.interval_ms
    }

    /// Flush the current batch and advance the flush timestamp.
    pub fn flush(&mut self, now_ms: i64) -> Vec<FillEvent> {
        self.last_flush_ms = now_ms;
        std::mem::take(&mut self.pending)
    }

    /// Current number of pending fills (for tests/diagnostics).
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Last flush timestamp (for coordination).
    pub fn last_flush_ms(&self) -> i64 {
        self.last_flush_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OrderPurpose, Side};
    use std::sync::Arc;

    fn mk_fill(venue_index: usize, price: f64, size: f64) -> FillEvent {
        FillEvent {
            venue_index,
            venue_id: Arc::from("test"),
            order_id: None,
            client_order_id: None,
            seq: None,
            side: Side::Buy,
            price,
            size,
            purpose: OrderPurpose::Mm,
            fee_bps: 0.0,
        }
    }

    #[test]
    fn batches_within_same_interval() {
        let mut batcher = FillBatcher::new(1_000);
        batcher.push(0, vec![mk_fill(0, 100.0, 1.0)]);
        batcher.push(0, vec![mk_fill(1, 101.0, 1.0)]);
        assert_eq!(batcher.pending_len(), 2);
        assert!(!batcher.should_flush(500));
        let flushed = batcher.flush(1_000);
        assert_eq!(flushed.len(), 2);
        assert_eq!(batcher.pending_len(), 0);
    }

    #[test]
    fn batches_across_intervals() {
        let mut batcher = FillBatcher::new(1_000);
        batcher.push(0, vec![mk_fill(0, 100.0, 1.0)]);
        let first = batcher.flush(1_000);
        assert_eq!(first.len(), 1);
        batcher.push(1_000, vec![mk_fill(1, 101.0, 1.0)]);
        let second = batcher.flush(2_000);
        assert_eq!(second.len(), 1);
    }
}
