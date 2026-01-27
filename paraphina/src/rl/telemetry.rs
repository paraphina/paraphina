// src/rl/telemetry.rs
//
// RL-specific telemetry for policy inputs/outputs and reward components.
//
// Per WHITEPAPER Appendix A.7, telemetry must include:
// - obs_version, policy_version, config_version
// - the policy action actually applied after safety clamps
// - per-tick reward components (or enough fields to reconstruct them)
// - episode boundary markers (start/end, termination reason)
//
// This extends the existing TelemetrySink with RL-specific fields.

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{self, Value as JsonValue};

use super::action_encoding::{encode_action, ACTION_VERSION};
use super::observation::{Observation, OBS_VERSION};
use super::policy::PolicyAction;
use super::research_budgets::{alignment_budget_for_profile, ResearchAlignmentBudget};
use super::runner::TerminationReason;
use super::safety::SafetyResult;
use crate::config::RiskProfile;

/// Reward components for per-tick logging and reward reconstruction.
///
/// Per WHITEPAPER Appendix A.4, includes:
/// - daily pnl change, dollar_delta_usd, basis_usd
/// - drawdown_abs, mean toxicity, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardComponents {
    /// Change in daily PnL since last tick.
    pub delta_pnl: f64,
    /// Absolute dollar delta (normalized by limit).
    pub delta_ratio: f64,
    /// Absolute basis exposure (normalized by limit).
    pub basis_ratio: f64,
    /// Absolute drawdown from peak.
    pub drawdown_abs: f64,
    /// Mean toxicity across venues.
    pub mean_toxicity: f64,
    /// Global inventory magnitude.
    pub inventory_abs: f64,
    /// Kill switch triggered this tick.
    pub kill_triggered: bool,
}

impl RewardComponents {
    /// Compute reward components from observation and previous state.
    pub fn from_observation(obs: &Observation, prev_pnl: f64, peak_pnl: f64) -> Self {
        let delta_ratio = if obs.delta_limit_usd > 0.0 {
            obs.dollar_delta_usd.abs() / obs.delta_limit_usd
        } else {
            0.0
        };

        let basis_ratio = if obs.basis_limit_hard_usd > 0.0 {
            obs.basis_usd.abs() / obs.basis_limit_hard_usd
        } else {
            0.0
        };

        let drawdown_abs = (peak_pnl - obs.daily_pnl_total).max(0.0);

        RewardComponents {
            delta_pnl: obs.daily_pnl_total - prev_pnl,
            delta_ratio,
            basis_ratio,
            drawdown_abs,
            mean_toxicity: obs.mean_toxicity(),
            inventory_abs: obs.q_global_tao.abs(),
            kill_triggered: obs.kill_switch,
        }
    }

    /// Compute a scalar reward using configurable weights.
    ///
    /// Per WHITEPAPER Appendix A.4:
    /// r_t = + Δ(daily_pnl_total)
    ///       - λ_delta * (|dollar_delta_usd| / delta_limit_usd)
    ///       - λ_basis * (|basis_usd| / basis_limit_hard_usd)
    ///       - λ_dd * max(0, drawdown_abs / dd_budget)
    ///       - λ_tox * mean_v(toxicity_v)
    ///       - λ_inv * |q_global_tao|
    pub fn compute_reward(&self, weights: &RewardWeights) -> f64 {
        let pnl_budget = weights.pnl_budget.max(1.0);
        let mut reward = self.delta_pnl / pnl_budget;
        reward -= weights.lambda_delta * self.delta_ratio;
        reward -= weights.lambda_basis * self.basis_ratio;
        reward -= weights.lambda_drawdown * (self.drawdown_abs / weights.drawdown_budget.max(1.0));
        reward -= weights.lambda_toxicity * self.mean_toxicity;
        reward -= weights.lambda_inventory * self.inventory_abs;

        if self.kill_triggered {
            reward += weights.kill_penalty; // negative penalty
        }

        reward
    }
}

/// Configurable weights for reward computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardWeights {
    /// Weight on delta exposure penalty.
    pub lambda_delta: f64,
    /// Weight on basis exposure penalty.
    pub lambda_basis: f64,
    /// Weight on drawdown penalty.
    pub lambda_drawdown: f64,
    /// Drawdown budget for normalization.
    pub drawdown_budget: f64,
    /// Mean PnL budget for normalization.
    pub pnl_budget: f64,
    /// Weight on mean toxicity.
    pub lambda_toxicity: f64,
    /// Weight on inventory magnitude.
    pub lambda_inventory: f64,
    /// Terminal penalty for kill switch trigger (negative).
    pub kill_penalty: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self::from_alignment_budget(RiskProfile::Balanced)
    }
}

impl RewardWeights {
    pub fn from_alignment_budget(profile: RiskProfile) -> Self {
        let budget = alignment_budget_for_profile(profile);
        Self::from_budget(profile, &budget)
    }

    pub fn from_budget(_profile: RiskProfile, budget: &ResearchAlignmentBudget) -> Self {
        let pnl_budget = budget.min_final_pnl_mean.abs().max(1.0);
        let kill_penalty = -1.0 / budget.max_kill_prob.max(1e-6);
        Self {
            lambda_delta: 1.0,
            lambda_basis: 0.5,
            lambda_drawdown: 2.0,
            drawdown_budget: budget.max_drawdown_abs,
            pnl_budget,
            lambda_toxicity: 0.1,
            lambda_inventory: 0.01,
            kill_penalty,
        }
    }
}

/// Per-tick record for RL telemetry logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickRecord {
    // ----- Metadata -----
    /// Observation schema version.
    pub obs_version: u32,
    /// Action encoding schema version.
    pub action_version: u32,
    /// Policy version string.
    pub policy_version: String,
    /// Config version string.
    pub config_version: String,
    /// Episode ID.
    pub episode_id: u64,
    /// Tick index within episode.
    pub tick_index: u64,
    /// Timestamp (ms).
    pub timestamp_ms: i64,

    // ----- Observation summary -----
    /// Fair value.
    pub fair_value: Option<f64>,
    /// Sigma effective.
    pub sigma_eff: f64,
    /// Global inventory (TAO).
    pub q_global_tao: f64,
    /// Dollar delta (USD).
    pub dollar_delta_usd: f64,
    /// Basis exposure (USD).
    pub basis_usd: f64,
    /// Daily PnL total.
    pub daily_pnl_total: f64,
    /// Risk regime.
    pub risk_regime: String,
    /// Kill switch status.
    pub kill_switch: bool,

    // ----- Policy action (post-clamp) -----
    /// Policy action applied this tick.
    pub action: Option<PolicyAction>,
    /// Raw policy action vector (pre-clamp).
    pub action_raw: Option<Vec<f64>>,
    /// Applied policy action vector (post-clamp).
    pub action_applied: Option<Vec<f64>>,
    /// Safety rejections / clamp reasons (if any).
    pub action_rejection_reasons: Option<Vec<String>>,
    /// Whether action was from shadow policy.
    pub is_shadow: bool,

    // ----- Reward components -----
    /// Reward components for this tick.
    pub reward_components: Option<RewardComponents>,
    /// Scalar reward (if computed).
    pub reward: Option<f64>,
}

impl TickRecord {
    pub fn new(
        obs: &Observation,
        action: Option<&PolicyAction>,
        config_version: &str,
        episode_id: u64,
    ) -> Self {
        Self {
            obs_version: OBS_VERSION,
            action_version: ACTION_VERSION,
            policy_version: action.map(|a| a.policy_version.clone()).unwrap_or_default(),
            config_version: config_version.to_string(),
            episode_id,
            tick_index: obs.tick_index,
            timestamp_ms: obs.timestamp_ms,
            fair_value: obs.fair_value,
            sigma_eff: obs.sigma_eff,
            q_global_tao: obs.q_global_tao,
            dollar_delta_usd: obs.dollar_delta_usd,
            basis_usd: obs.basis_usd,
            daily_pnl_total: obs.daily_pnl_total,
            risk_regime: format!("{:?}", obs.risk_regime),
            kill_switch: obs.kill_switch,
            action: action.cloned(),
            action_raw: None,
            action_applied: None,
            action_rejection_reasons: None,
            is_shadow: false,
            reward_components: None,
            reward: None,
        }
    }

    pub fn with_safety(mut self, safety: Option<&SafetyResult>, num_venues: usize) -> Self {
        if let Some(result) = safety {
            let raw = encode_action(&result.raw, num_venues)
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            let applied = encode_action(&result.applied, num_venues)
                .into_iter()
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            self.action_raw = Some(raw);
            self.action_applied = Some(applied);
            if !result.rejection_reasons.is_empty() {
                self.action_rejection_reasons = Some(result.rejection_reasons.clone());
            }
        }
        self
    }

    pub fn with_reward(mut self, components: RewardComponents, weights: &RewardWeights) -> Self {
        let reward = components.compute_reward(weights);
        self.reward_components = Some(components);
        self.reward = Some(reward);
        self
    }

    pub fn with_shadow(mut self, is_shadow: bool) -> Self {
        self.is_shadow = is_shadow;
        self
    }
}

/// Episode marker for logging episode boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMarker {
    /// Episode ID.
    pub episode_id: u64,
    /// Random seed used for this episode.
    pub seed: u64,
    /// Marker type.
    pub marker_type: EpisodeMarkerType,
    /// Timestamp (ms).
    pub timestamp_ms: i64,
    /// Termination reason (for end markers).
    pub termination_reason: Option<TerminationReason>,
    /// Final PnL (for end markers).
    pub final_pnl: Option<f64>,
    /// Total ticks in episode (for end markers).
    pub total_ticks: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EpisodeMarkerType {
    Start,
    End,
}

/// RL-specific telemetry sink.
///
/// Extends the existing telemetry infrastructure with RL-specific logging.
/// Controlled by environment variables:
/// - PARAPHINA_RL_TELEMETRY_MODE: "off" (default) or "jsonl"
/// - PARAPHINA_RL_TELEMETRY_PATH: path to JSONL file
pub struct RLTelemetry {
    enabled: bool,
    path: Option<PathBuf>,
    writer: Option<BufWriter<File>>,
    episode_id: u64,
    prev_pnl: f64,
    peak_pnl: f64,
    reward_weights: RewardWeights,
}

impl Default for RLTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl RLTelemetry {
    /// Create a new RL telemetry sink (disabled by default).
    pub fn new() -> Self {
        Self {
            enabled: false,
            path: None,
            writer: None,
            episode_id: 0,
            prev_pnl: 0.0,
            peak_pnl: 0.0,
            reward_weights: RewardWeights::default(),
        }
    }

    /// Create from environment variables.
    pub fn from_env() -> Self {
        let enabled = env::var("PARAPHINA_RL_TELEMETRY_MODE")
            .map(|s| s.to_lowercase() == "jsonl")
            .unwrap_or(false);

        let path = env::var("PARAPHINA_RL_TELEMETRY_PATH")
            .ok()
            .map(PathBuf::from);

        Self {
            enabled,
            path,
            writer: None,
            episode_id: 0,
            prev_pnl: 0.0,
            peak_pnl: 0.0,
            reward_weights: RewardWeights::default(),
        }
    }

    /// Enable telemetry with a specific path.
    pub fn enable(path: PathBuf) -> Self {
        Self {
            enabled: true,
            path: Some(path),
            writer: None,
            episode_id: 0,
            prev_pnl: 0.0,
            peak_pnl: 0.0,
            reward_weights: RewardWeights::default(),
        }
    }

    /// Set custom reward weights.
    pub fn with_reward_weights(mut self, weights: RewardWeights) -> Self {
        self.reward_weights = weights;
        self
    }

    /// Reset for a new episode.
    pub fn reset_episode(&mut self, episode_id: u64, _seed: u64) {
        self.episode_id = episode_id;
        self.prev_pnl = 0.0;
        self.peak_pnl = 0.0;
    }

    fn ensure_writer(&mut self) -> Option<&mut BufWriter<File>> {
        if !self.enabled {
            return None;
        }

        if self.writer.is_none() {
            let path = self.path.as_ref()?;

            // Create parent directories
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()?;

            self.writer = Some(BufWriter::new(file));
        }

        self.writer.as_mut()
    }

    fn write_json(&mut self, value: &JsonValue) {
        let Some(writer) = self.ensure_writer() else {
            return;
        };

        let line = match serde_json::to_string(value) {
            Ok(s) => s,
            Err(_) => return,
        };

        if writeln!(writer, "{}", line).is_err() {
            self.enabled = false;
            self.writer = None;
        }
    }

    /// Log episode start marker.
    pub fn log_episode_start(&mut self, episode_id: u64, seed: u64, timestamp_ms: i64) {
        let marker = EpisodeMarker {
            episode_id,
            seed,
            marker_type: EpisodeMarkerType::Start,
            timestamp_ms,
            termination_reason: None,
            final_pnl: None,
            total_ticks: None,
        };

        let value = serde_json::to_value(&marker).unwrap_or_default();
        self.write_json(&value);
    }

    /// Log episode end marker.
    pub fn log_episode_end(
        &mut self,
        episode_id: u64,
        seed: u64,
        timestamp_ms: i64,
        reason: TerminationReason,
        final_pnl: f64,
        total_ticks: u64,
    ) {
        let marker = EpisodeMarker {
            episode_id,
            seed,
            marker_type: EpisodeMarkerType::End,
            timestamp_ms,
            termination_reason: Some(reason),
            final_pnl: Some(final_pnl),
            total_ticks: Some(total_ticks),
        };

        let value = serde_json::to_value(&marker).unwrap_or_default();
        self.write_json(&value);
    }

    /// Log a tick record with reward components.
    pub fn log_tick(
        &mut self,
        obs: &Observation,
        action: Option<&PolicyAction>,
        safety: Option<&SafetyResult>,
        config_version: &str,
    ) {
        // Update peak PnL
        if obs.daily_pnl_total > self.peak_pnl {
            self.peak_pnl = obs.daily_pnl_total;
        }

        // Compute reward components
        let components = RewardComponents::from_observation(obs, self.prev_pnl, self.peak_pnl);

        let record = TickRecord::new(obs, action, config_version, self.episode_id)
            .with_safety(safety, obs.venues.len())
            .with_reward(components, &self.reward_weights);

        // Update prev_pnl for next tick
        self.prev_pnl = obs.daily_pnl_total;

        let value = serde_json::to_value(&record).unwrap_or_default();
        self.write_json(&value);
    }

    /// Log a shadow policy tick (proposals not executed).
    pub fn log_shadow_tick(
        &mut self,
        obs: &Observation,
        action: &PolicyAction,
        safety: Option<&SafetyResult>,
        config_version: &str,
    ) {
        let record = TickRecord::new(obs, Some(action), config_version, self.episode_id)
            .with_safety(safety, obs.venues.len())
            .with_shadow(true);

        let value = serde_json::to_value(&record).unwrap_or_default();
        self.write_json(&value);
    }

    /// Flush the writer.
    pub fn flush(&mut self) {
        if let Some(writer) = &mut self.writer {
            let _ = writer.flush();
        }
    }

    /// Check if telemetry is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Drop for RLTelemetry {
    fn drop(&mut self) {
        self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::rl::observation::Observation;
    use crate::state::GlobalState;

    fn create_test_observation(pnl: f64, tick_index: u64) -> Observation {
        let cfg = Config::default();
        let mut state = GlobalState::new(&cfg);
        state.fair_value = Some(300.0);
        state.sigma_eff = 0.02;
        state.daily_pnl_total = pnl;
        state.daily_realised_pnl = pnl;
        state.dollar_delta_usd = 5000.0;
        state.delta_limit_usd = 100_000.0;
        state.basis_usd = 500.0;
        state.basis_limit_hard_usd = 10_000.0;
        Observation::from_state(&state, &cfg, 1000 + tick_index as i64 * 1000, tick_index)
    }

    #[test]
    fn test_reward_components_from_observation() {
        let obs = create_test_observation(100.0, 0);
        let components = RewardComponents::from_observation(&obs, 0.0, 100.0);

        assert!((components.delta_pnl - 100.0).abs() < 1e-9);
        assert!((components.delta_ratio - 0.05).abs() < 1e-9); // 5000/100000
        assert!((components.basis_ratio - 0.05).abs() < 1e-9); // 500/10000
        assert!((components.drawdown_abs - 0.0).abs() < 1e-9); // no drawdown
    }

    #[test]
    fn test_reward_computation() {
        let obs = create_test_observation(100.0, 0);
        let components = RewardComponents::from_observation(&obs, 0.0, 100.0);
        let weights = RewardWeights::default();

        let reward = components.compute_reward(&weights);

        // Reward should include PnL change (100) minus penalties
        assert!(reward < 100.0, "Penalties should reduce reward");
        assert!(
            reward > 0.0,
            "Reward should still be positive with small exposures"
        );
    }

    #[test]
    fn test_tick_record_creation() {
        let obs = create_test_observation(100.0, 5);
        let action = PolicyAction::identity(5, "test-v1");
        let record = TickRecord::new(&obs, Some(&action), "config-v1", 1);

        assert_eq!(record.tick_index, 5);
        assert_eq!(record.episode_id, 1);
        assert_eq!(record.policy_version, "test-v1");
        assert_eq!(record.config_version, "config-v1");
    }

    #[test]
    fn test_tick_record_serialization() {
        let obs = create_test_observation(100.0, 5);
        let action = PolicyAction::identity(5, "test-v1");
        let record = TickRecord::new(&obs, Some(&action), "config-v1", 1);

        let json = serde_json::to_string(&record).unwrap();
        let parsed: TickRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.tick_index, parsed.tick_index);
        assert_eq!(record.episode_id, parsed.episode_id);
    }

    #[test]
    fn test_episode_marker_serialization() {
        let marker = EpisodeMarker {
            episode_id: 42,
            seed: 12345,
            marker_type: EpisodeMarkerType::End,
            timestamp_ms: 1000,
            termination_reason: Some(TerminationReason::EndOfEpisode),
            final_pnl: Some(500.0),
            total_ticks: Some(100),
        };

        let json = serde_json::to_string(&marker).unwrap();
        let parsed: EpisodeMarker = serde_json::from_str(&json).unwrap();

        assert_eq!(marker.episode_id, parsed.episode_id);
        assert_eq!(marker.seed, parsed.seed);
    }
}
