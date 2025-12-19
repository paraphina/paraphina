// src/rl/policy.rs
//
// Policy trait and implementations for RL-0.
//
// Per WHITEPAPER Appendix A.2, the policy operates on a bounded "control surface"
// rather than raw prices/orders. This keeps the action space small, continuous,
// and bounded for stable RL training.
//
// Design:
// - Policy trait: defines interface for all policy implementations
// - PolicyAction: bounded knobs that modulate the heuristic baseline
// - HeuristicPolicy: preserves current behavior (identity transformation)
// - NoopPolicy: proposes no changes (for shadow mode baseline)

use serde::{Deserialize, Serialize};

use super::observation::Observation;

/// Current heuristic policy version.
pub const HEURISTIC_POLICY_VERSION: &str = "heuristic-v1.0.0";

/// Policy action: bounded control surface for policy outputs.
///
/// Per WHITEPAPER Appendix A.2, these are bounded "knobs" that modulate
/// the existing quoting + hedging logic, rather than raw prices/orders.
///
/// All values are multiplicative modifiers around 1.0 (neutral).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyAction {
    /// Policy version identifier.
    pub policy_version: String,

    /// Optional policy ID for tracking (e.g., model checkpoint name).
    pub policy_id: Option<String>,

    // ----- Per-venue quote modifiers -----
    /// Spread scale per venue: multiplies computed half-spread.
    /// Range: [0.5, 3.0], default 1.0.
    pub spread_scale: Vec<f64>,

    /// Size scale per venue: multiplies computed size.
    /// Range: [0.0, 2.0], default 1.0.
    pub size_scale: Vec<f64>,

    /// Reservation price offset per venue (USD).
    /// Range: [-R, +R], default 0.0.
    pub rprice_offset_usd: Vec<f64>,

    // ----- Global hedge modifiers -----
    /// Hedge scale: multiplies desired hedge step.
    /// Range: [0.0, 2.0], default 1.0.
    pub hedge_scale: f64,

    /// Hedge venue weights: allocation preference across hedge-allowed venues.
    /// Should sum to 1.0 (simplex), but will be normalized if not.
    /// Default: equal weights.
    pub hedge_venue_weights: Vec<f64>,
}

impl PolicyAction {
    /// Create a neutral/identity policy action for a given number of venues.
    ///
    /// This represents "do nothing different from baseline".
    pub fn identity(num_venues: usize, policy_version: &str) -> Self {
        Self {
            policy_version: policy_version.to_string(),
            policy_id: None,
            spread_scale: vec![1.0; num_venues],
            size_scale: vec![1.0; num_venues],
            rprice_offset_usd: vec![0.0; num_venues],
            hedge_scale: 1.0,
            hedge_venue_weights: vec![1.0 / num_venues as f64; num_venues],
        }
    }

    /// Clamp all values to their valid ranges.
    pub fn clamp(&mut self) {
        for v in &mut self.spread_scale {
            *v = v.clamp(0.5, 3.0);
        }
        for v in &mut self.size_scale {
            *v = v.clamp(0.0, 2.0);
        }
        // rprice_offset_usd typically bounded by config, but use a reasonable default
        const MAX_OFFSET_USD: f64 = 10.0;
        for v in &mut self.rprice_offset_usd {
            *v = v.clamp(-MAX_OFFSET_USD, MAX_OFFSET_USD);
        }
        self.hedge_scale = self.hedge_scale.clamp(0.0, 2.0);

        // Normalize hedge venue weights to simplex
        let sum: f64 = self.hedge_venue_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.hedge_venue_weights {
                *w /= sum;
            }
        }
    }

    /// Check if this action is neutral (identity transformation).
    pub fn is_identity(&self) -> bool {
        let eps = 1e-6;

        let spread_neutral = self.spread_scale.iter().all(|&v| (v - 1.0).abs() < eps);
        let size_neutral = self.size_scale.iter().all(|&v| (v - 1.0).abs() < eps);
        let offset_neutral = self.rprice_offset_usd.iter().all(|&v| v.abs() < eps);
        let hedge_neutral = (self.hedge_scale - 1.0).abs() < eps;

        spread_neutral && size_neutral && offset_neutral && hedge_neutral
    }
}

/// Policy trait: interface for all policy implementations.
///
/// Policies map Observations to PolicyActions. The safety layer then
/// applies clamps and validates before execution.
pub trait Policy: Send + Sync {
    /// Unique version string for this policy implementation.
    fn version(&self) -> &str;

    /// Optional policy ID (e.g., model checkpoint name).
    fn policy_id(&self) -> Option<&str> {
        None
    }

    /// Compute a policy action given the current observation.
    ///
    /// This should be a pure function: same observation → same action.
    fn act(&self, obs: &Observation) -> PolicyAction;

    /// Reset the policy for a new episode.
    ///
    /// Called at the start of each episode to reset any internal state.
    /// The seed enables deterministic episode sequences.
    fn reset_episode(&mut self, seed: u64, episode_id: u64);
}

/// Heuristic policy: preserves current baseline behavior.
///
/// This is the default policy that applies identity transformations,
/// meaning the existing MM/hedge logic runs unmodified.
pub struct HeuristicPolicy {
    version: String,
    policy_id: Option<String>,
    /// Current episode seed (for determinism).
    _seed: u64,
    /// Current episode ID.
    _episode_id: u64,
}

impl Default for HeuristicPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicPolicy {
    pub fn new() -> Self {
        Self {
            version: HEURISTIC_POLICY_VERSION.to_string(),
            policy_id: Some("baseline".to_string()),
            _seed: 0,
            _episode_id: 0,
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.policy_id = Some(id.to_string());
        self
    }
}

impl Policy for HeuristicPolicy {
    fn version(&self) -> &str {
        &self.version
    }

    fn policy_id(&self) -> Option<&str> {
        self.policy_id.as_deref()
    }

    fn act(&self, obs: &Observation) -> PolicyAction {
        // Identity transformation: don't modify anything from baseline
        PolicyAction::identity(obs.venues.len(), &self.version)
    }

    fn reset_episode(&mut self, seed: u64, episode_id: u64) {
        self._seed = seed;
        self._episode_id = episode_id;
    }
}

/// Noop policy: proposes no changes (useful for shadow mode validation).
///
/// Unlike HeuristicPolicy, this explicitly signals "do nothing".
pub struct NoopPolicy {
    version: String,
}

impl Default for NoopPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl NoopPolicy {
    pub fn new() -> Self {
        Self {
            version: "noop-v1.0.0".to_string(),
        }
    }
}

impl Policy for NoopPolicy {
    fn version(&self) -> &str {
        &self.version
    }

    fn policy_id(&self) -> Option<&str> {
        Some("noop")
    }

    fn act(&self, obs: &Observation) -> PolicyAction {
        PolicyAction::identity(obs.venues.len(), &self.version)
    }

    fn reset_episode(&mut self, _seed: u64, _episode_id: u64) {
        // Noop has no state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::rl::observation::Observation;
    use crate::state::GlobalState;

    fn create_test_observation() -> Observation {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(300.0);
        state.sigma_eff = 0.02;
        Observation::from_state(&state, &cfg, 1000, 0)
    }

    #[test]
    fn test_policy_action_identity() {
        let action = PolicyAction::identity(5, "test-v1");

        assert!(action.is_identity());
        assert_eq!(action.spread_scale.len(), 5);
        assert_eq!(action.size_scale.len(), 5);
        assert_eq!(action.rprice_offset_usd.len(), 5);
    }

    #[test]
    fn test_policy_action_clamp() {
        let mut action = PolicyAction::identity(3, "test-v1");

        // Set out-of-range values
        action.spread_scale = vec![0.1, 5.0, 1.5];
        action.size_scale = vec![-1.0, 3.0, 1.0];
        action.hedge_scale = 5.0;

        action.clamp();

        assert_eq!(action.spread_scale, vec![0.5, 3.0, 1.5]);
        assert_eq!(action.size_scale, vec![0.0, 2.0, 1.0]);
        assert_eq!(action.hedge_scale, 2.0);
    }

    #[test]
    fn test_heuristic_policy_deterministic() {
        let obs = create_test_observation();

        let policy = HeuristicPolicy::new();
        let action1 = policy.act(&obs);
        let action2 = policy.act(&obs);

        assert_eq!(
            action1, action2,
            "Same observation should produce same action"
        );
    }

    #[test]
    fn test_heuristic_policy_is_identity() {
        let obs = create_test_observation();
        let policy = HeuristicPolicy::new();
        let action = policy.act(&obs);

        assert!(action.is_identity());
    }

    #[test]
    fn test_noop_policy() {
        let obs = create_test_observation();
        let policy = NoopPolicy::new();
        let action = policy.act(&obs);

        assert!(action.is_identity());
        assert_eq!(policy.version(), "noop-v1.0.0");
    }

    #[test]
    fn test_policy_reset_episode() {
        let mut policy = HeuristicPolicy::new();
        policy.reset_episode(42, 1);

        // Should not panic or change behavior
        let obs = create_test_observation();
        let action = policy.act(&obs);
        assert!(action.is_identity());
    }

    #[test]
    fn test_policy_action_serialization() {
        let action = PolicyAction::identity(5, "test-v1");
        let json = serde_json::to_string(&action).unwrap();
        let parsed: PolicyAction = serde_json::from_str(&json).unwrap();

        assert_eq!(action, parsed);
    }
}
