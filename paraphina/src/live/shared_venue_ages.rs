//! Shared venue age tracking for cross-task health monitoring.
//!
//! Provides a lock-free, `Arc`-wrapped structure that the runner updates
//! each tick and that Layer A (enforcer) and Layer B (REST monitor) read.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

/// Thread-safe, lock-free venue age tracking.
///
/// The runner updates `ages[venue_index]` each tick with the current
/// `age_ms = now_ms - last_mid_apply_ms`.  Readers (health enforcer,
/// REST monitor) can load these values at any time with `Relaxed`
/// ordering — stale reads by a few ms are acceptable.
#[derive(Clone)]
pub struct SharedVenueAges {
    ages: Arc<Vec<AtomicI64>>,
    /// Monotonic timestamp (ms since epoch) of the runner's last write.
    /// Used by Layer A to detect a stuck runner.
    last_write_ms: Arc<AtomicI64>,
}

impl std::fmt::Debug for SharedVenueAges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedVenueAges")
            .field("len", &self.ages.len())
            .finish()
    }
}

impl SharedVenueAges {
    /// Create a new instance with `count` venues, all initialised to `i64::MAX`
    /// (infinitely stale).
    pub fn new(count: usize) -> Self {
        let ages: Vec<AtomicI64> = (0..count).map(|_| AtomicI64::new(i64::MAX)).collect();
        Self {
            ages: Arc::new(ages),
            last_write_ms: Arc::new(AtomicI64::new(0)),
        }
    }

    /// Update the age for a single venue.  Called by the runner each tick.
    #[inline]
    pub fn set_age(&self, venue_index: usize, age_ms: i64) {
        if let Some(slot) = self.ages.get(venue_index) {
            // Map the telemetry sentinel -1 (no data ever) to i64::MAX.
            let val = if age_ms < 0 { i64::MAX } else { age_ms };
            slot.store(val, Ordering::Relaxed);
        }
    }

    /// Mark the runner as alive.  Called after all ages are written.
    #[inline]
    pub fn mark_write(&self, now_ms: i64) {
        self.last_write_ms.store(now_ms, Ordering::Relaxed);
    }

    /// Read the current age for a venue.  Returns `i64::MAX` if the venue
    /// index is out of range or has never been updated.
    #[inline]
    pub fn age_ms(&self, venue_index: usize) -> i64 {
        self.ages
            .get(venue_index)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(i64::MAX)
    }

    /// Milliseconds since the runner last called [`mark_write`].
    /// Returns `i64::MAX` if the runner has never written.
    #[inline]
    pub fn runner_idle_ms(&self, now_ms: i64) -> i64 {
        let last = self.last_write_ms.load(Ordering::Relaxed);
        if last == 0 {
            i64::MAX
        } else {
            (now_ms - last).max(0)
        }
    }

    /// Number of tracked venues.
    #[inline]
    pub fn len(&self) -> usize {
        self.ages.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_max() {
        let ages = SharedVenueAges::new(3);
        assert_eq!(ages.age_ms(0), i64::MAX);
        assert_eq!(ages.age_ms(1), i64::MAX);
        assert_eq!(ages.age_ms(2), i64::MAX);
        assert_eq!(ages.runner_idle_ms(1000), i64::MAX);
    }

    #[test]
    fn set_and_read() {
        let ages = SharedVenueAges::new(2);
        ages.set_age(0, 42);
        ages.set_age(1, 100);
        ages.mark_write(5000);
        assert_eq!(ages.age_ms(0), 42);
        assert_eq!(ages.age_ms(1), 100);
        assert_eq!(ages.runner_idle_ms(5050), 50);
    }

    #[test]
    fn negative_age_mapped_to_max() {
        let ages = SharedVenueAges::new(1);
        ages.set_age(0, -1);
        assert_eq!(ages.age_ms(0), i64::MAX);
    }

    #[test]
    fn out_of_range_returns_max() {
        let ages = SharedVenueAges::new(1);
        assert_eq!(ages.age_ms(99), i64::MAX);
    }
}
