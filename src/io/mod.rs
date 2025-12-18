// src/io/mod.rs
//
// I/O boundary layer for Paraphina (Milestone H).
//
// This module defines the execution gateway abstraction that separates
// the pure strategy core from I/O operations. The strategy emits Actions;
// the gateway executes them on venues.
//
// Design principles:
// - VenueAdapter trait abstracts venue-specific execution
// - Gateway orchestrates multiple adapters and applies policies
// - Rate limiting and retry policies are configurable
// - Adapters can be swapped for sim/live/replay modes

pub mod noop;
pub mod sim;

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::actions::{Action, ActionBatch, CancelAllAction, CancelOrderAction, PlaceOrderAction};
use crate::config::Config;
use crate::state::GlobalState;
use crate::types::{FillEvent, TimestampMs};

/// Result of executing an action on a venue.
#[derive(Debug, Clone)]
pub enum ActionResult {
    /// Order was filled (fully or partially).
    Filled(FillEvent),
    /// Order was placed but not yet filled (resting).
    Placed { order_id: String },
    /// Order was cancelled.
    Cancelled { order_id: String },
    /// All orders cancelled (with count).
    CancelledAll { count: usize },
    /// Action acknowledged but no fill (e.g., kill switch).
    Acknowledged,
    /// Action was rejected.
    Rejected { reason: String },
    /// Action failed due to error.
    Error { message: String },
}

/// Trait for venue-specific execution adapters.
///
/// Implementations handle the venue-specific logic for executing actions.
/// - SimAdapter: Simulates fills using fee models
/// - NoopAdapter: Records actions without executing (for replay)
/// - LiveAdapter: (future) Real exchange API calls
pub trait VenueAdapter: Send + Sync {
    /// Execute a PlaceOrder action.
    fn place_order(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        action: &PlaceOrderAction,
    ) -> ActionResult;

    /// Execute a CancelOrder action.
    fn cancel_order(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        action: &CancelOrderAction,
    ) -> ActionResult;

    /// Execute a CancelAll action.
    fn cancel_all(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        action: &CancelAllAction,
    ) -> ActionResult;

    /// Get the adapter name for logging.
    fn name(&self) -> &str;
}

/// Rate limiting policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitPolicy {
    /// Maximum requests per second.
    pub max_requests_per_second: f64,
    /// Burst capacity (requests allowed above rate limit).
    pub burst_capacity: u32,
    /// Whether rate limiting is enabled.
    pub enabled: bool,
}

impl Default for RateLimitPolicy {
    fn default() -> Self {
        Self {
            max_requests_per_second: 10.0,
            burst_capacity: 5,
            enabled: false, // Disabled by default for sim
        }
    }
}

/// Retry policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial backoff duration.
    pub initial_backoff: Duration,
    /// Maximum backoff duration.
    pub max_backoff: Duration,
    /// Backoff multiplier (exponential backoff).
    pub backoff_multiplier: f64,
    /// Whether retries are enabled.
    pub enabled: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            enabled: false, // Disabled by default for sim
        }
    }
}

/// Combined gateway policy for I/O operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GatewayPolicy {
    /// Rate limiting configuration.
    pub rate_limit: RateLimitPolicy,
    /// Retry configuration.
    pub retry: RetryPolicy,
}

impl GatewayPolicy {
    /// Create a policy for simulation (no rate limiting, no retries).
    pub fn for_simulation() -> Self {
        Self::default()
    }

    /// Create a policy for live trading with sensible defaults.
    pub fn for_live() -> Self {
        Self {
            rate_limit: RateLimitPolicy {
                max_requests_per_second: 5.0,
                burst_capacity: 10,
                enabled: true,
            },
            retry: RetryPolicy {
                max_retries: 3,
                initial_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                enabled: true,
            },
        }
    }

    /// Create from environment variables.
    ///
    /// Environment variables:
    /// - PARAPHINA_RATE_LIMIT_RPS: max requests per second
    /// - PARAPHINA_RATE_LIMIT_BURST: burst capacity
    /// - PARAPHINA_RATE_LIMIT_ENABLED: enable rate limiting
    /// - PARAPHINA_RETRY_MAX: max retry attempts
    /// - PARAPHINA_RETRY_BACKOFF_MS: initial backoff in ms
    /// - PARAPHINA_RETRY_ENABLED: enable retries
    pub fn from_env() -> Self {
        use std::env;

        let mut policy = Self::default();

        if let Ok(val) = env::var("PARAPHINA_RATE_LIMIT_RPS") {
            if let Ok(rps) = val.parse::<f64>() {
                policy.rate_limit.max_requests_per_second = rps.max(0.1);
            }
        }

        if let Ok(val) = env::var("PARAPHINA_RATE_LIMIT_BURST") {
            if let Ok(burst) = val.parse::<u32>() {
                policy.rate_limit.burst_capacity = burst;
            }
        }

        if let Ok(val) = env::var("PARAPHINA_RATE_LIMIT_ENABLED") {
            policy.rate_limit.enabled = val.to_lowercase() == "true" || val == "1";
        }

        if let Ok(val) = env::var("PARAPHINA_RETRY_MAX") {
            if let Ok(max) = val.parse::<u32>() {
                policy.retry.max_retries = max;
            }
        }

        if let Ok(val) = env::var("PARAPHINA_RETRY_BACKOFF_MS") {
            if let Ok(ms) = val.parse::<u64>() {
                policy.retry.initial_backoff = Duration::from_millis(ms);
            }
        }

        if let Ok(val) = env::var("PARAPHINA_RETRY_ENABLED") {
            policy.retry.enabled = val.to_lowercase() == "true" || val == "1";
        }

        policy
    }
}

/// Simple rate limiter state.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    policy: RateLimitPolicy,
    tokens: f64,
    last_update_ms: TimestampMs,
}

impl RateLimiter {
    /// Create a new rate limiter with the given policy.
    pub fn new(policy: RateLimitPolicy) -> Self {
        Self {
            tokens: policy.burst_capacity as f64,
            last_update_ms: -1, // Use -1 to indicate uninitialized
            policy,
        }
    }

    /// Check if a request is allowed and consume a token if so.
    pub fn try_acquire(&mut self, now_ms: TimestampMs) -> bool {
        if !self.policy.enabled {
            return true;
        }

        // Replenish tokens based on time elapsed (if not first call)
        if self.last_update_ms >= 0 && now_ms > self.last_update_ms {
            let elapsed_sec = (now_ms - self.last_update_ms) as f64 / 1000.0;
            let replenished = elapsed_sec * self.policy.max_requests_per_second;
            self.tokens = (self.tokens + replenished).min(self.policy.burst_capacity as f64);
        }
        self.last_update_ms = now_ms;

        // Check if we have a token
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Result of executing an action batch.
#[derive(Debug, Clone)]
pub struct BatchExecutionResult {
    /// Results for each action in the batch.
    pub results: Vec<(Action, ActionResult)>,
    /// Fills generated from this batch.
    pub fills: Vec<FillEvent>,
    /// Number of successful actions.
    pub success_count: usize,
    /// Number of failed actions.
    pub failure_count: usize,
}

/// Execution gateway that orchestrates venue adapters.
///
/// The gateway owns adapters for each venue and applies policies
/// (rate limiting, retries) when executing actions.
pub struct Gateway {
    /// Venue adapters indexed by venue_index.
    adapters: Vec<Box<dyn VenueAdapter>>,
    /// Gateway policy for rate limiting and retries.
    policy: GatewayPolicy,
    /// Per-venue rate limiters.
    rate_limiters: Vec<RateLimiter>,
}

impl Gateway {
    /// Create a new gateway with the given adapters and policy.
    pub fn new(adapters: Vec<Box<dyn VenueAdapter>>, policy: GatewayPolicy) -> Self {
        let rate_limiters = adapters
            .iter()
            .map(|_| RateLimiter::new(policy.rate_limit.clone()))
            .collect();

        Self {
            adapters,
            policy,
            rate_limiters,
        }
    }

    /// Execute a batch of actions.
    pub fn execute_batch(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        batch: &ActionBatch,
        now_ms: TimestampMs,
    ) -> BatchExecutionResult {
        let mut results = Vec::new();
        let mut fills = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        for action in &batch.actions {
            let result = self.execute_action(cfg, state, action, now_ms);

            // Collect fills
            if let ActionResult::Filled(ref fill) = result {
                fills.push(fill.clone());
            }

            // Track success/failure
            match &result {
                ActionResult::Filled(_)
                | ActionResult::Placed { .. }
                | ActionResult::Cancelled { .. }
                | ActionResult::CancelledAll { .. }
                | ActionResult::Acknowledged => {
                    success_count += 1;
                }
                ActionResult::Rejected { .. } | ActionResult::Error { .. } => {
                    failure_count += 1;
                }
            }

            results.push((action.clone(), result));
        }

        BatchExecutionResult {
            results,
            fills,
            success_count,
            failure_count,
        }
    }

    /// Execute a single action.
    fn execute_action(
        &mut self,
        cfg: &Config,
        state: &mut GlobalState,
        action: &Action,
        now_ms: TimestampMs,
    ) -> ActionResult {
        match action {
            Action::PlaceOrder(po) => {
                // Check rate limit
                if let Some(rl) = self.rate_limiters.get_mut(po.venue_index) {
                    if !rl.try_acquire(now_ms) {
                        return ActionResult::Rejected {
                            reason: "Rate limited".to_string(),
                        };
                    }
                }

                // Execute on adapter
                if let Some(adapter) = self.adapters.get_mut(po.venue_index) {
                    adapter.place_order(cfg, state, po)
                } else {
                    ActionResult::Error {
                        message: format!("No adapter for venue {}", po.venue_index),
                    }
                }
            }

            Action::CancelOrder(co) => {
                if let Some(adapter) = self.adapters.get_mut(co.venue_index) {
                    adapter.cancel_order(cfg, state, co)
                } else {
                    ActionResult::Error {
                        message: format!("No adapter for venue {}", co.venue_index),
                    }
                }
            }

            Action::CancelAll(ca) => {
                if let Some(venue_idx) = ca.venue_index {
                    if let Some(adapter) = self.adapters.get_mut(venue_idx) {
                        adapter.cancel_all(cfg, state, ca)
                    } else {
                        ActionResult::Error {
                            message: format!("No adapter for venue {}", venue_idx),
                        }
                    }
                } else {
                    // Global cancel all
                    let mut total_count = 0;
                    for adapter in &mut self.adapters {
                        if let ActionResult::CancelledAll { count } =
                            adapter.cancel_all(cfg, state, ca)
                        {
                            total_count += count;
                        }
                    }
                    ActionResult::CancelledAll { count: total_count }
                }
            }

            Action::SetKillSwitch(_) => {
                // Kill switch is handled by the strategy runner, not the gateway
                // But we acknowledge it here
                ActionResult::Acknowledged
            }

            Action::Log(_) => {
                // Log actions are no-ops at the gateway level
                ActionResult::Acknowledged
            }
        }
    }

    /// Get a reference to the policy.
    pub fn policy(&self) -> &GatewayPolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_disabled() {
        let policy = RateLimitPolicy {
            enabled: false,
            ..Default::default()
        };
        let mut limiter = RateLimiter::new(policy);

        // Should always allow when disabled
        for _ in 0..100 {
            assert!(limiter.try_acquire(1000));
        }
    }

    #[test]
    fn test_rate_limiter_enabled() {
        let policy = RateLimitPolicy {
            max_requests_per_second: 10.0,
            burst_capacity: 3,
            enabled: true,
        };
        let mut limiter = RateLimiter::new(policy);

        // Should allow burst
        assert!(limiter.try_acquire(0));
        assert!(limiter.try_acquire(0));
        assert!(limiter.try_acquire(0));

        // Should deny after burst exhausted
        assert!(!limiter.try_acquire(0));

        // After 100ms, should replenish 1 token (0.1s * 10 rps = 1)
        assert!(limiter.try_acquire(100));

        // Immediately after using that token, should deny
        assert!(!limiter.try_acquire(100));

        // After another 200ms total (from 100 to 300), replenish 2 tokens
        // 0.2s * 10 rps = 2 tokens
        assert!(limiter.try_acquire(300));
        assert!(limiter.try_acquire(300));
        assert!(!limiter.try_acquire(300));
    }

    #[test]
    fn test_gateway_policy_default() {
        let policy = GatewayPolicy::default();
        assert!(!policy.rate_limit.enabled);
        assert!(!policy.retry.enabled);
    }

    #[test]
    fn test_gateway_policy_for_live() {
        let policy = GatewayPolicy::for_live();
        assert!(policy.rate_limit.enabled);
        assert!(policy.retry.enabled);
    }
}
