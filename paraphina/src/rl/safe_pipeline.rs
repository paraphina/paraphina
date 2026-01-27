// src/rl/safe_pipeline.rs
//
// Deterministic safe-RL smoke pipeline using policy surfaces + safety layer.

use serde::{Deserialize, Serialize};

use crate::config::Config;

use super::action_encoding::decode_action;
use super::policy::PolicyAction;
use super::sim_env::{SimEnv, SimEnvConfig};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafePipelineSummary {
    pub seed: u64,
    pub episodes: u64,
    pub ticks_per_episode: u64,
    pub total_steps: u64,
    pub total_reward: f64,
    pub safety_rejection_events: u64,
    pub max_applied_spread_scale: f64,
    pub max_applied_size_scale: f64,
    pub max_applied_rprice_offset_usd: f64,
    pub max_applied_hedge_scale: f64,
    pub min_applied_hedge_weight: f64,
    pub max_applied_hedge_weight: f64,
    pub reward_drawdown_budget: f64,
    pub reward_kill_penalty: f64,
}

fn build_unsafe_action(num_venues: usize) -> PolicyAction {
    let mut action = PolicyAction::identity(num_venues, "rl-safe-pipeline");
    action.spread_scale = vec![10.0; num_venues];
    action.size_scale = vec![5.0; num_venues];
    action.rprice_offset_usd = vec![100.0; num_venues];
    action.hedge_scale = 10.0;
    action.hedge_venue_weights = vec![0.0; num_venues];
    action
}

pub fn run_safe_pipeline(
    base_config: &Config,
    seed: u64,
    episodes: u64,
    ticks_per_episode: u64,
) -> SafePipelineSummary {
    let mut env_config = SimEnvConfig::deterministic();
    env_config.max_ticks = ticks_per_episode;

    let reward_drawdown_budget = env_config.reward_weights.drawdown_budget;
    let reward_kill_penalty = env_config.reward_weights.kill_penalty;

    let mut env = SimEnv::new(base_config.clone(), env_config);
    let num_venues = env.num_venues().max(1);

    let mut total_steps = 0u64;
    let mut total_reward = 0.0;
    let mut safety_rejection_events = 0u64;
    let mut max_spread: f64 = 0.0;
    let mut max_size: f64 = 0.0;
    let mut max_offset: f64 = 0.0;
    let mut max_hedge_scale: f64 = 0.0;
    let mut min_weight: f64 = 1.0;
    let mut max_weight: f64 = 0.0;

    for episode in 0..episodes {
        env.reset(Some(seed + episode));
        for _ in 0..ticks_per_episode {
            let action = build_unsafe_action(num_venues);
            let result = env.step(&action);
            total_steps += 1;
            total_reward += result.reward;

            if let Some(reasons) = &result.info.policy_rejection_reasons {
                if !reasons.is_empty() {
                    safety_rejection_events += 1;
                }
            }

            if let Some(vec) = &result.info.policy_action_applied {
                let applied_vec: Vec<f32> = vec.iter().map(|v| *v as f32).collect();
                let applied = decode_action(&applied_vec, num_venues, "rl-safe-applied");
                if let Some(v) = applied
                    .spread_scale
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b))
                {
                    max_spread = max_spread.max(v);
                }
                if let Some(v) = applied
                    .size_scale
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b))
                {
                    max_size = max_size.max(v);
                }
                if let Some(v) = applied
                    .rprice_offset_usd
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.total_cmp(b))
                {
                    max_offset = max_offset.max(v);
                }
                max_hedge_scale = max_hedge_scale.max(applied.hedge_scale);
                if let Some(v) = applied
                    .hedge_venue_weights
                    .iter()
                    .copied()
                    .min_by(|a, b| a.total_cmp(b))
                {
                    min_weight = min_weight.min(v);
                }
                if let Some(v) = applied
                    .hedge_venue_weights
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b))
                {
                    max_weight = max_weight.max(v);
                }
            }

            if result.done {
                break;
            }
        }
    }

    SafePipelineSummary {
        seed,
        episodes,
        ticks_per_episode,
        total_steps,
        total_reward,
        safety_rejection_events,
        max_applied_spread_scale: max_spread,
        max_applied_size_scale: max_size,
        max_applied_rprice_offset_usd: max_offset,
        max_applied_hedge_scale: max_hedge_scale,
        min_applied_hedge_weight: min_weight,
        max_applied_hedge_weight: max_weight,
        reward_drawdown_budget,
        reward_kill_penalty,
    }
}
