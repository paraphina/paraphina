// src/rl/trajectory.rs
//
// RL-2: Trajectory collection for behaviour cloning.
//
// Per ROADMAP.md RL-2, this module provides:
// - Deterministic trajectory collection from the HeuristicPolicy
// - Support for vectorised rollouts (N envs in parallel)
// - Output format: binary records + JSON metadata
//
// Design:
// - TrajectoryRecord: single (obs, action, reward, terminal, info) transition
// - TrajectoryCollector: runs rollouts and collects trajectories
// - TrajectoryWriter: serializes to files with versioned metadata

use std::io::{self, Write};

use serde::{Deserialize, Serialize};

use super::action_encoding::{encode_action, ActionEncodingSpec, ACTION_VERSION};
use super::observation::{Observation, OBS_VERSION};
use super::policy::{HeuristicPolicy, Policy, PolicyAction, HEURISTIC_POLICY_VERSION};
use super::sim_env::{SimEnvConfig, VecEnv};
use crate::config::Config;

/// Current trajectory format version.
/// Increment when changing the record schema.
pub const TRAJECTORY_VERSION: u32 = 1;

/// Observation encoding for compact storage.
///
/// Converts Observation to a flat f32 vector for neural network input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsEncoding {
    /// Global features (17 values):
    /// fair_value, sigma_eff, vol_ratio_clipped, spread_mult, size_mult, band_mult,
    /// q_global_tao, dollar_delta_usd, delta_limit_usd,
    /// basis_usd, basis_gross_usd, basis_limit_warn_usd, basis_limit_hard_usd,
    /// daily_realised_pnl, daily_unrealised_pnl, daily_pnl_total,
    /// kill_switch (0/1)
    pub global_features: Vec<f32>,
    /// Per-venue features (14 per venue):
    /// mid, spread, depth_near_mid, staleness_ms,
    /// local_vol_short, local_vol_long, status (one-hot 3), toxicity,
    /// position_tao, avg_entry_price, funding_8h, margin_available_usd, dist_liq_sigma
    pub venue_features: Vec<f32>,
}

/// Number of global observation features.
pub const OBS_GLOBAL_DIM: usize = 17;

/// Number of features per venue.
/// mid, spread, depth_near_mid, staleness_ms, local_vol_short, local_vol_long,
/// status (3 one-hot: Healthy, Warning, Disabled), toxicity, position_tao,
/// avg_entry_price, funding_8h, margin_available_usd, dist_liq_sigma = 15
pub const OBS_PER_VENUE_DIM: usize = 15;

impl ObsEncoding {
    /// Encode an Observation to a flat vector.
    pub fn from_observation(obs: &Observation) -> Self {
        let mut global = Vec::with_capacity(OBS_GLOBAL_DIM);

        // Global features
        global.push(obs.fair_value.unwrap_or(0.0) as f32);
        global.push(obs.sigma_eff as f32);
        global.push(obs.vol_ratio_clipped as f32);
        global.push(obs.spread_mult as f32);
        global.push(obs.size_mult as f32);
        global.push(obs.band_mult as f32);
        global.push(obs.q_global_tao as f32);
        global.push(obs.dollar_delta_usd as f32);
        global.push(obs.delta_limit_usd as f32);
        global.push(obs.basis_usd as f32);
        global.push(obs.basis_gross_usd as f32);
        global.push(obs.basis_limit_warn_usd as f32);
        global.push(obs.basis_limit_hard_usd as f32);
        global.push(obs.daily_realised_pnl as f32);
        global.push(obs.daily_unrealised_pnl as f32);
        global.push(obs.daily_pnl_total as f32);
        global.push(if obs.kill_switch { 1.0 } else { 0.0 });

        // Per-venue features
        let num_venues = obs.venues.len();
        let mut venue_features = Vec::with_capacity(num_venues * OBS_PER_VENUE_DIM);

        for v in &obs.venues {
            venue_features.push(v.mid.unwrap_or(0.0) as f32);
            venue_features.push(v.spread.unwrap_or(0.0) as f32);
            venue_features.push(v.depth_near_mid as f32);
            venue_features.push(v.staleness_ms.unwrap_or(0) as f32);
            venue_features.push(v.local_vol_short as f32);
            venue_features.push(v.local_vol_long as f32);

            // Status one-hot: [Healthy, Warning, Disabled]
            use super::observation::VenueStatusObs;
            let (h, w, d) = match v.status {
                VenueStatusObs::Healthy => (1.0, 0.0, 0.0),
                VenueStatusObs::Warning => (0.0, 1.0, 0.0),
                VenueStatusObs::Disabled => (0.0, 0.0, 1.0),
            };
            venue_features.push(h);
            venue_features.push(w);
            venue_features.push(d);

            venue_features.push(v.toxicity as f32);
            venue_features.push(v.position_tao as f32);
            venue_features.push(v.avg_entry_price as f32);
            venue_features.push(v.funding_8h as f32);
            venue_features.push(v.margin_available_usd as f32);
            venue_features.push(v.dist_liq_sigma as f32);
        }

        Self {
            global_features: global,
            venue_features,
        }
    }

    /// Get the total observation dimension for a given number of venues.
    pub fn dim(num_venues: usize) -> usize {
        OBS_GLOBAL_DIM + num_venues * OBS_PER_VENUE_DIM
    }

    /// Flatten to a single vector.
    pub fn to_flat_vec(&self) -> Vec<f32> {
        let mut flat = self.global_features.clone();
        flat.extend(&self.venue_features);
        flat
    }
}

/// A single trajectory record (transition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    /// Observation features (flat vector).
    pub obs_features: Vec<f32>,
    /// Action target (normalized vector from HeuristicPolicy).
    pub action_target: Vec<f32>,
    /// Reward for this transition.
    pub reward: f32,
    /// Whether this is a terminal state.
    pub terminal: bool,
    /// Kill reason if terminal due to kill switch.
    pub kill_reason: Option<String>,
    /// Episode index.
    pub episode_idx: u32,
    /// Step index within episode.
    pub step_idx: u32,
    /// Seed used for this episode.
    pub seed: u64,
}

/// Metadata for a trajectory dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMetadata {
    /// Trajectory format version.
    pub trajectory_version: u32,
    /// Observation schema version.
    pub obs_version: u32,
    /// Action encoding version.
    pub action_version: u32,
    /// Policy version used for collection.
    pub policy_version: String,
    /// Base random seed.
    pub base_seed: u64,
    /// Number of episodes collected.
    pub num_episodes: u32,
    /// Total number of transitions.
    pub num_transitions: u64,
    /// Number of venues.
    pub num_venues: usize,
    /// Observation dimension.
    pub obs_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Domain randomisation preset used.
    pub domain_rand_preset: String,
    /// Apply domain randomisation flag.
    pub apply_domain_rand: bool,
    /// Maximum ticks per episode.
    pub max_ticks: u64,
    /// Collection timestamp (ISO 8601).
    pub collected_at: String,
    /// Kill rate (fraction of episodes ending in kill).
    pub kill_rate: f64,
    /// Mean episode length.
    pub mean_episode_length: f64,
    /// Mean total PnL.
    pub mean_pnl: f64,
}

/// Collects trajectories from HeuristicPolicy rollouts.
pub struct TrajectoryCollector {
    /// Number of parallel environments.
    num_envs: usize,
    /// Maximum ticks per episode.
    max_ticks: u64,
    /// Apply domain randomisation.
    apply_domain_rand: bool,
    /// Domain randomisation preset.
    domain_rand_preset: String,
    /// Base seed for reproducibility.
    base_seed: u64,
}

impl TrajectoryCollector {
    /// Create a new trajectory collector.
    pub fn new(
        num_envs: usize,
        max_ticks: u64,
        apply_domain_rand: bool,
        domain_rand_preset: &str,
        base_seed: u64,
    ) -> Self {
        Self {
            num_envs,
            max_ticks,
            apply_domain_rand,
            domain_rand_preset: domain_rand_preset.to_string(),
            base_seed,
        }
    }

    /// Collect trajectories for the specified number of episodes.
    ///
    /// Returns (records, metadata).
    pub fn collect(&self, num_episodes: u32) -> (Vec<TrajectoryRecord>, TrajectoryMetadata) {
        let base_config = Config::default();
        let num_venues = base_config.venues.len();

        let domain_rand = match self.domain_rand_preset.as_str() {
            "default" => super::domain_rand::DomainRandConfig::default(),
            "mild" => super::domain_rand::DomainRandConfig::mild(),
            "deterministic" => super::domain_rand::DomainRandConfig::deterministic(),
            _ => super::domain_rand::DomainRandConfig::default(),
        };

        let env_config = SimEnvConfig {
            max_ticks: self.max_ticks,
            dt_ms: 1000,
            domain_rand,
            reward_weights: Default::default(),
            apply_domain_rand: self.apply_domain_rand,
            ablations: Default::default(),
        };

        let mut vec_env = VecEnv::new(self.num_envs, base_config.clone(), env_config);

        let policy = HeuristicPolicy::new();

        let mut records = Vec::new();
        let mut episode_idx = 0_u32;
        let mut total_kills = 0_u32;
        let mut total_steps = 0_u64;
        let mut total_pnl = 0.0_f64;
        let mut episode_lengths = Vec::new();

        let action_spec = ActionEncodingSpec::new(num_venues);

        while episode_idx < num_episodes {
            // Generate seeds for this batch
            let batch_size = (num_episodes - episode_idx).min(self.num_envs as u32) as usize;
            let seeds: Vec<u64> = (0..batch_size)
                .map(|i| {
                    self.base_seed
                        .wrapping_add((episode_idx as u64 + i as u64) * 12345)
                })
                .collect();

            // Extend seeds to full vec_env size
            let full_seeds: Vec<u64> = (0..self.num_envs)
                .map(|i| {
                    if i < seeds.len() {
                        seeds[i]
                    } else {
                        self.base_seed.wrapping_add(u64::MAX - i as u64)
                    }
                })
                .collect();

            let observations = vec_env.reset_all(Some(&full_seeds));

            // Track per-env state
            let env_episode_idx: Vec<u32> = (0..self.num_envs)
                .map(|i| {
                    if i < batch_size {
                        episode_idx + i as u32
                    } else {
                        u32::MAX // Placeholder for unused envs
                    }
                })
                .collect();
            let mut env_step_idx: Vec<u32> = vec![0; self.num_envs];
            let mut env_done: Vec<bool> = (0..self.num_envs).map(|i| i >= batch_size).collect();
            let env_seeds = full_seeds;

            // Store initial observations for first step
            let mut prev_observations = observations;

            // Run episodes
            loop {
                // Check if all tracked episodes are done
                let active_count = env_done
                    .iter()
                    .enumerate()
                    .filter(|(i, d)| !**d && env_episode_idx[*i] < num_episodes)
                    .count();
                if active_count == 0 {
                    break;
                }

                // Compute actions using heuristic policy
                let actions: Vec<PolicyAction> = prev_observations
                    .iter()
                    .map(|obs| policy.act(obs))
                    .collect();

                // Step all environments
                let results = vec_env.step(&actions);

                // Record transitions for active environments
                for (i, result) in results.iter().enumerate() {
                    if env_done[i] || env_episode_idx[i] >= num_episodes {
                        continue;
                    }

                    let obs_enc = ObsEncoding::from_observation(&prev_observations[i]);
                    let action_enc = encode_action(&actions[i], num_venues);

                    let kill_reason = if result.info.kill_switch {
                        result.info.kill_reason.clone()
                    } else {
                        None
                    };

                    records.push(TrajectoryRecord {
                        obs_features: obs_enc.to_flat_vec(),
                        action_target: action_enc,
                        reward: result.reward as f32,
                        terminal: result.done,
                        kill_reason,
                        episode_idx: env_episode_idx[i],
                        step_idx: env_step_idx[i],
                        seed: env_seeds[i],
                    });

                    env_step_idx[i] += 1;
                    total_steps += 1;

                    if result.done {
                        env_done[i] = true;
                        total_pnl += result.info.pnl_total;
                        episode_lengths.push(env_step_idx[i] as u64);

                        if result.info.kill_switch {
                            total_kills += 1;
                        }
                    }
                }

                // Update previous observations
                prev_observations = results.iter().map(|r| r.observation.clone()).collect();
            }

            episode_idx += batch_size as u32;
        }

        let kill_rate = if num_episodes > 0 {
            total_kills as f64 / num_episodes as f64
        } else {
            0.0
        };

        let mean_episode_length = if !episode_lengths.is_empty() {
            episode_lengths.iter().sum::<u64>() as f64 / episode_lengths.len() as f64
        } else {
            0.0
        };

        let mean_pnl = if num_episodes > 0 {
            total_pnl / num_episodes as f64
        } else {
            0.0
        };

        let metadata = TrajectoryMetadata {
            trajectory_version: TRAJECTORY_VERSION,
            obs_version: OBS_VERSION,
            action_version: ACTION_VERSION,
            policy_version: HEURISTIC_POLICY_VERSION.to_string(),
            base_seed: self.base_seed,
            num_episodes,
            num_transitions: total_steps,
            num_venues,
            obs_dim: ObsEncoding::dim(num_venues),
            action_dim: action_spec.action_dim,
            domain_rand_preset: self.domain_rand_preset.clone(),
            apply_domain_rand: self.apply_domain_rand,
            max_ticks: self.max_ticks,
            collected_at: chrono_now_iso8601(),
            kill_rate,
            mean_episode_length,
            mean_pnl,
        };

        (records, metadata)
    }
}

/// Get current timestamp in ISO 8601 format.
fn chrono_now_iso8601() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Writes trajectory data to files.
pub struct TrajectoryWriter {
    /// Output directory path.
    output_dir: String,
}

impl TrajectoryWriter {
    /// Create a new writer with the given output directory.
    pub fn new(output_dir: &str) -> Self {
        Self {
            output_dir: output_dir.to_string(),
        }
    }

    /// Write records and metadata to files.
    ///
    /// Creates:
    /// - {output_dir}/trajectories.bin - binary records
    /// - {output_dir}/metadata.json - JSON metadata
    pub fn write(
        &self,
        records: &[TrajectoryRecord],
        metadata: &TrajectoryMetadata,
    ) -> io::Result<()> {
        use std::fs;

        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // Write metadata as JSON
        let metadata_path = format!("{}/metadata.json", self.output_dir);
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&metadata_path, metadata_json)?;

        // Write records as JSON Lines (simple, portable format)
        let records_path = format!("{}/trajectories.jsonl", self.output_dir);
        let mut file = fs::File::create(&records_path)?;

        for record in records {
            let line = serde_json::to_string(record)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{}", line)?;
        }

        Ok(())
    }

    /// Write records as separate NPZ-compatible arrays.
    ///
    /// Creates individual .npy-style files that can be loaded by numpy.
    pub fn write_arrays(
        &self,
        records: &[TrajectoryRecord],
        metadata: &TrajectoryMetadata,
    ) -> io::Result<()> {
        use std::fs;

        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // Write metadata
        let metadata_path = format!("{}/metadata.json", self.output_dir);
        let metadata_json = serde_json::to_string_pretty(metadata)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&metadata_path, metadata_json)?;

        // Extract arrays
        let n = records.len();
        let obs_dim = metadata.obs_dim;
        let action_dim = metadata.action_dim;

        // Flatten observations
        let mut obs_flat: Vec<f32> = Vec::with_capacity(n * obs_dim);
        let mut action_flat: Vec<f32> = Vec::with_capacity(n * action_dim);
        let mut rewards: Vec<f32> = Vec::with_capacity(n);
        let mut terminals: Vec<u8> = Vec::with_capacity(n);
        let mut episode_indices: Vec<u32> = Vec::with_capacity(n);

        for record in records {
            obs_flat.extend(&record.obs_features);
            action_flat.extend(&record.action_target);
            rewards.push(record.reward);
            terminals.push(if record.terminal { 1 } else { 0 });
            episode_indices.push(record.episode_idx);
        }

        // Write as raw binary with shape header
        Self::write_array_f32(
            &format!("{}/observations.bin", self.output_dir),
            &obs_flat,
            &[n as u64, obs_dim as u64],
        )?;
        Self::write_array_f32(
            &format!("{}/actions.bin", self.output_dir),
            &action_flat,
            &[n as u64, action_dim as u64],
        )?;
        Self::write_array_f32(
            &format!("{}/rewards.bin", self.output_dir),
            &rewards,
            &[n as u64],
        )?;
        Self::write_array_u8(
            &format!("{}/terminals.bin", self.output_dir),
            &terminals,
            &[n as u64],
        )?;
        Self::write_array_u32(
            &format!("{}/episode_indices.bin", self.output_dir),
            &episode_indices,
            &[n as u64],
        )?;

        // Write shape info as JSON for easy loading
        let shapes = serde_json::json!({
            "observations": [n, obs_dim],
            "actions": [n, action_dim],
            "rewards": [n],
            "terminals": [n],
            "episode_indices": [n],
        });
        let shapes_path = format!("{}/shapes.json", self.output_dir);
        fs::write(&shapes_path, shapes.to_string())?;

        Ok(())
    }

    /// Write a f32 array with shape header.
    fn write_array_f32(path: &str, data: &[f32], shape: &[u64]) -> io::Result<()> {
        use std::fs::File;

        let mut file = File::create(path)?;

        // Write shape as header (number of dims, then each dim)
        let ndims = shape.len() as u64;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in shape {
            file.write_all(&dim.to_le_bytes())?;
        }

        // Write data
        for &val in data {
            file.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }

    /// Write a u8 array with shape header.
    fn write_array_u8(path: &str, data: &[u8], shape: &[u64]) -> io::Result<()> {
        use std::fs::File;

        let mut file = File::create(path)?;

        // Write shape as header
        let ndims = shape.len() as u64;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in shape {
            file.write_all(&dim.to_le_bytes())?;
        }

        // Write data
        file.write_all(data)?;

        Ok(())
    }

    /// Write a u32 array with shape header.
    fn write_array_u32(path: &str, data: &[u32], shape: &[u64]) -> io::Result<()> {
        use std::fs::File;

        let mut file = File::create(path)?;

        // Write shape as header
        let ndims = shape.len() as u64;
        file.write_all(&ndims.to_le_bytes())?;
        for &dim in shape {
            file.write_all(&dim.to_le_bytes())?;
        }

        // Write data
        for &val in data {
            file.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obs_encoding_dim() {
        let num_venues = 5;
        let expected_dim = OBS_GLOBAL_DIM + num_venues * OBS_PER_VENUE_DIM;
        assert_eq!(ObsEncoding::dim(num_venues), expected_dim);
    }

    #[test]
    fn test_obs_encoding_from_observation() {
        let cfg = Config::default();
        let mut state = crate::state::GlobalState::new(&cfg);

        state.fair_value = Some(300.0);
        state.sigma_eff = 0.02;
        state.spread_mult = 1.0;

        for v in &mut state.venues {
            v.mid = Some(300.0);
            v.spread = Some(1.0);
        }

        let obs = Observation::from_state(&state, &cfg, 1000, 0);
        let enc = ObsEncoding::from_observation(&obs);

        assert_eq!(enc.global_features.len(), OBS_GLOBAL_DIM);
        assert_eq!(
            enc.venue_features.len(),
            cfg.venues.len() * OBS_PER_VENUE_DIM
        );

        let flat = enc.to_flat_vec();
        assert_eq!(flat.len(), ObsEncoding::dim(cfg.venues.len()));
    }

    #[test]
    fn test_trajectory_collector_small() {
        let collector = TrajectoryCollector::new(2, 10, false, "deterministic", 42);
        let (records, metadata) = collector.collect(2);

        assert!(!records.is_empty());
        assert_eq!(metadata.num_episodes, 2);
        assert_eq!(metadata.trajectory_version, TRAJECTORY_VERSION);
        assert_eq!(metadata.obs_version, OBS_VERSION);
        assert_eq!(metadata.action_version, ACTION_VERSION);
    }

    #[test]
    fn test_trajectory_collector_determinism() {
        let collector1 = TrajectoryCollector::new(2, 10, false, "deterministic", 42);
        let (records1, _) = collector1.collect(2);

        let collector2 = TrajectoryCollector::new(2, 10, false, "deterministic", 42);
        let (records2, _) = collector2.collect(2);

        // Same seed should produce same records
        assert_eq!(records1.len(), records2.len());

        for (r1, r2) in records1.iter().zip(records2.iter()) {
            assert_eq!(r1.obs_features, r2.obs_features);
            assert_eq!(r1.action_target, r2.action_target);
            assert_eq!(r1.reward, r2.reward);
            assert_eq!(r1.terminal, r2.terminal);
            assert_eq!(r1.episode_idx, r2.episode_idx);
            assert_eq!(r1.step_idx, r2.step_idx);
        }
    }

    #[test]
    fn test_trajectory_record_serialization() {
        let record = TrajectoryRecord {
            obs_features: vec![1.0, 2.0, 3.0],
            action_target: vec![0.5, 0.5],
            reward: 1.0,
            terminal: false,
            kill_reason: None,
            episode_idx: 0,
            step_idx: 0,
            seed: 42,
        };

        let json = serde_json::to_string(&record).unwrap();
        let parsed: TrajectoryRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.obs_features, parsed.obs_features);
        assert_eq!(record.action_target, parsed.action_target);
        assert_eq!(record.reward, parsed.reward);
    }
}
