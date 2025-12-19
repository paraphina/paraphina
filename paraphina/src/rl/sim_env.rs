// src/rl/sim_env.rs
//
// Gym-style simulation environment for RL-1 training.
//
// Per ROADMAP.md RL-1, this module provides:
// - SimEnv: Single Gym-style environment (reset, step)
// - VecEnv: Vectorised environments for parallel rollouts
// - Deterministic execution given seeds
//
// The environment wraps the existing simulation engine without duplicating logic.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::engine::Engine;
use crate::exit;
use crate::gateway::{ExecutionGateway, SimGateway};
use crate::hedge::{compute_hedge_plan, hedge_plan_to_order_intents};
use crate::mm::{compute_mm_quotes, mm_quotes_to_order_intents};
use crate::state::{GlobalState, PendingMarkoutRecord};
use crate::types::{FillEvent, OrderIntent, Side, TimestampMs};

use super::domain_rand::{DomainRandConfig, DomainRandSample, DomainRandSampler};
use super::observation::Observation;
use super::policy::PolicyAction;
use super::telemetry::{RewardComponents, RewardWeights};

/// Result of a single environment step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// The observation after taking the action.
    pub observation: Observation,
    /// The reward for this step.
    pub reward: f64,
    /// Whether the episode has terminated.
    pub done: bool,
    /// Additional information about the step.
    pub info: StepInfo,
}

/// Additional information returned from a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepInfo {
    /// Termination reason if done.
    pub termination_reason: Option<String>,
    /// Current tick index.
    pub tick: u64,
    /// Total PnL.
    pub pnl_total: f64,
    /// Realised PnL.
    pub pnl_realised: f64,
    /// Unrealised PnL.
    pub pnl_unrealised: f64,
    /// Whether kill switch was triggered.
    pub kill_switch: bool,
    /// Kill reason if any.
    pub kill_reason: Option<String>,
    /// Risk regime as string.
    pub risk_regime: String,
    /// Global inventory (TAO).
    pub q_global_tao: f64,
    /// Dollar delta (USD).
    pub dollar_delta_usd: f64,
    /// Basis exposure (USD).
    pub basis_usd: f64,
    /// Domain randomisation sample used this episode.
    pub domain_rand_sample: Option<DomainRandSample>,
    /// Reward components breakdown.
    pub reward_components: Option<RewardComponents>,
}

impl Default for StepInfo {
    fn default() -> Self {
        Self {
            termination_reason: None,
            tick: 0,
            pnl_total: 0.0,
            pnl_realised: 0.0,
            pnl_unrealised: 0.0,
            kill_switch: false,
            kill_reason: None,
            risk_regime: "Normal".to_string(),
            q_global_tao: 0.0,
            dollar_delta_usd: 0.0,
            basis_usd: 0.0,
            domain_rand_sample: None,
            reward_components: None,
        }
    }
}

/// Configuration for the simulation environment.
#[derive(Debug, Clone)]
pub struct SimEnvConfig {
    /// Maximum number of ticks per episode.
    pub max_ticks: u64,
    /// Time delta per tick in milliseconds.
    pub dt_ms: i64,
    /// Domain randomisation configuration.
    pub domain_rand: DomainRandConfig,
    /// Reward weights for computing scalar reward.
    pub reward_weights: RewardWeights,
    /// Whether to apply domain randomisation.
    pub apply_domain_rand: bool,
}

impl Default for SimEnvConfig {
    fn default() -> Self {
        Self {
            max_ticks: 1000,
            dt_ms: 1000,
            domain_rand: DomainRandConfig::default(),
            reward_weights: RewardWeights::default(),
            apply_domain_rand: true,
        }
    }
}

impl SimEnvConfig {
    /// Create a config with no randomisation (for deterministic tests).
    pub fn deterministic() -> Self {
        Self {
            max_ticks: 1000,
            dt_ms: 1000,
            domain_rand: DomainRandConfig::deterministic(),
            reward_weights: RewardWeights::default(),
            apply_domain_rand: false,
        }
    }

    /// Create a config with mild randomisation.
    pub fn mild_rand() -> Self {
        Self {
            max_ticks: 1000,
            dt_ms: 1000,
            domain_rand: DomainRandConfig::mild(),
            reward_weights: RewardWeights::default(),
            apply_domain_rand: true,
        }
    }
}

/// Gym-style simulation environment.
///
/// Wraps the Paraphina simulator to provide a standard RL interface:
/// - reset(seed) -> observation
/// - step(action) -> (observation, reward, done, info)
///
/// All state transitions are deterministic given the seed.
pub struct SimEnv {
    /// Base configuration.
    base_config: Config,
    /// Environment configuration.
    env_config: SimEnvConfig,
    /// Current configuration (possibly modified by domain rand).
    config: Config,
    /// Engine instance (reserved for future use).
    #[allow(dead_code)]
    engine: Option<Engine<'static>>,
    /// Current state.
    state: GlobalState,
    /// Execution gateway.
    gateway: SimGateway,
    /// Domain randomisation sampler.
    domain_rand_sampler: Option<DomainRandSampler>,
    /// Current domain rand sample.
    current_domain_rand: Option<DomainRandSample>,
    /// Random number generator for simulation noise.
    rng: ChaCha8Rng,
    /// Current tick index.
    tick: u64,
    /// Base timestamp for this episode.
    base_ms: TimestampMs,
    /// Previous PnL for reward computation.
    prev_pnl: f64,
    /// Peak PnL for drawdown computation.
    peak_pnl: f64,
    /// Whether the episode is done.
    done: bool,
    /// Current seed.
    seed: u64,
    /// Static config holder (for engine lifetime).
    static_config: Option<Box<Config>>,
}

impl SimEnv {
    /// Create a new simulation environment.
    pub fn new(base_config: Config, env_config: SimEnvConfig) -> Self {
        let state = GlobalState::new(&base_config);
        let domain_rand_sampler = if env_config.apply_domain_rand {
            Some(DomainRandSampler::new(env_config.domain_rand.clone(), 0))
        } else {
            None
        };

        Self {
            config: base_config.clone(),
            base_config,
            env_config,
            engine: None,
            state,
            gateway: SimGateway::new(),
            domain_rand_sampler,
            current_domain_rand: None,
            rng: ChaCha8Rng::seed_from_u64(0),
            tick: 0,
            base_ms: 0,
            prev_pnl: 0.0,
            peak_pnl: 0.0,
            done: false,
            seed: 0,
            static_config: None,
        }
    }

    /// Reset the environment with an optional seed.
    ///
    /// Returns the initial observation.
    pub fn reset(&mut self, seed: Option<u64>) -> Observation {
        // Use provided seed or generate from RNG
        let seed = seed.unwrap_or_else(|| self.rng.gen());
        self.seed = seed;

        // Reseed the RNG
        self.rng = ChaCha8Rng::seed_from_u64(seed);

        // Apply domain randomisation if enabled
        if let Some(ref mut sampler) = self.domain_rand_sampler {
            sampler.reseed(seed);
            let sample = sampler.sample_episode(self.base_config.venues.len());
            self.apply_domain_rand_sample(&sample);
            self.current_domain_rand = Some(sample);
        } else {
            self.config = self.base_config.clone();
            self.current_domain_rand = None;
        }

        // Store config in a box for static lifetime
        self.static_config = Some(Box::new(self.config.clone()));

        // Reset state
        self.state = GlobalState::new(&self.config);

        // Reset episode state
        self.tick = 0;
        self.base_ms = (seed % 10_000) as i64;
        self.prev_pnl = 0.0;
        self.peak_pnl = 0.0;
        self.done = false;

        // Apply initial position from domain rand or config
        self.apply_initial_position();

        // Initial engine tick to set up fair value
        let now_ms = self.base_ms;
        self.seed_synthetic_data(now_ms);
        self.run_engine_tick(now_ms);

        // Build and return initial observation
        Observation::from_state(&self.state, &self.config, now_ms, 0)
    }

    /// Take a step in the environment.
    ///
    /// The action is a PolicyAction that modulates the baseline strategy.
    pub fn step(&mut self, action: &PolicyAction) -> StepResult {
        if self.done {
            // Return terminal observation
            let now_ms = self.base_ms + (self.tick as i64) * self.env_config.dt_ms;
            let obs = Observation::from_state(&self.state, &self.config, now_ms, self.tick);
            return StepResult {
                observation: obs,
                reward: 0.0,
                done: true,
                info: self.build_step_info(Some("Episode already done".to_string())),
            };
        }

        // Advance tick
        self.tick += 1;
        let now_ms = self.base_ms + (self.tick as i64) * self.env_config.dt_ms;

        // Seed synthetic market data
        self.seed_synthetic_data(now_ms);

        // Run engine tick (FV, vol, toxicity, risk)
        self.run_engine_tick(now_ms);

        // Execute strategy with the given action
        self.execute_strategy(now_ms, action);

        // Check termination conditions
        let termination_reason = self.check_termination();
        self.done = termination_reason.is_some();

        // Compute reward
        let (reward, reward_components) = self.compute_reward();

        // Update PnL tracking
        self.prev_pnl = self.state.daily_pnl_total;
        if self.state.daily_pnl_total > self.peak_pnl {
            self.peak_pnl = self.state.daily_pnl_total;
        }

        // Build observation
        let obs = Observation::from_state(&self.state, &self.config, now_ms, self.tick);

        // Build info
        let mut info = self.build_step_info(termination_reason);
        info.reward_components = Some(reward_components);

        StepResult {
            observation: obs,
            reward,
            done: self.done,
            info,
        }
    }

    /// Get the number of venues.
    pub fn num_venues(&self) -> usize {
        self.config.venues.len()
    }

    /// Get current state (for testing).
    pub fn state(&self) -> &GlobalState {
        &self.state
    }

    /// Get current seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Get current tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Check if episode is done.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Apply domain randomisation sample to the config.
    fn apply_domain_rand_sample(&mut self, sample: &DomainRandSample) {
        self.config = self.base_config.clone();

        // Apply fee randomisation
        for (i, vcfg) in self.config.venues.iter_mut().enumerate() {
            if i < sample.maker_fees_bps.len() {
                vcfg.maker_fee_bps = sample.maker_fees_bps[i];
            }
            if i < sample.taker_fees_bps.len() {
                vcfg.taker_fee_bps = sample.taker_fees_bps[i];
            }
            if i < sample.maker_rebates_bps.len() {
                vcfg.maker_rebate_bps = sample.maker_rebates_bps[i];
            }
        }

        // Apply volatility randomisation
        self.config.volatility.vol_ref = sample.vol_ref * sample.vol_multiplier;
        self.config.volatility.fv_vol_alpha_short = sample.vol_alpha_short;
        self.config.volatility.fv_vol_alpha_long = sample.vol_alpha_long;

        // Apply slippage randomisation
        self.config.exit.slippage_linear_coeff = sample.slippage_linear;
        self.config.exit.slippage_quadratic_coeff = sample.slippage_quadratic;

        // Apply initial inventory
        self.config.initial_q_tao = sample.initial_q_tao;
    }

    /// Apply initial position from config.
    fn apply_initial_position(&mut self) {
        let q0 = self.config.initial_q_tao;
        if q0.abs() < 1e-9 {
            return;
        }

        let venue_index = 0;
        let size_tao = q0.abs();
        let side = if q0 > 0.0 { Side::Buy } else { Side::Sell };

        // Use initial fair value from domain rand or a default
        let fair = self
            .current_domain_rand
            .as_ref()
            .map(|s| s.initial_fair_value)
            .unwrap_or(300.0);

        self.state
            .apply_perp_fill(venue_index, side, size_tao, fair, 0.0);
        self.state.recompute_after_fills(&self.config);
    }

    /// Seed synthetic market data for simulation.
    fn seed_synthetic_data(&mut self, now_ms: TimestampMs) {
        let sample = self.current_domain_rand.as_ref();
        let base = sample.map(|s| s.initial_fair_value).unwrap_or(300.0);

        // Add some randomness to the base price
        let price_noise: f64 = self.rng.gen_range(-0.5..0.5);
        let base = base + (now_ms as f64) * 0.0001 + price_noise;

        let spread_mult = sample.map(|s| s.spread_mult).unwrap_or(1.0);

        for (idx, v) in self.state.venues.iter_mut().enumerate() {
            let offset = idx as f64 * 0.4;
            let mid_prev = v.mid;

            let mid = base + offset;
            let spread_offset = sample
                .and_then(|s| s.spread_offsets.get(idx))
                .copied()
                .unwrap_or(0.0);
            let spread = (0.4 + idx as f64 * 0.02) * spread_mult + spread_offset;
            let depth = sample
                .and_then(|s| s.depth_near_mid.get(idx))
                .copied()
                .unwrap_or(10_000.0);

            // Apply funding from domain rand
            let funding = sample
                .and_then(|s| s.funding_8h.get(idx))
                .copied()
                .unwrap_or(0.0);
            v.funding_8h = funding;

            // Update order-book snapshot.
            v.mid = Some(mid);
            v.spread = Some(spread.max(0.01));
            v.last_mid_update_ms = Some(now_ms);
            v.depth_near_mid = depth;

            // Update local per-venue volatility
            if let Some(prev) = mid_prev {
                if prev > 0.0 && mid > 0.0 {
                    let r = (mid / prev).ln();
                    let r2 = r * r;

                    let alpha_short = self.config.volatility.fv_vol_alpha_short;
                    let alpha_long = self.config.volatility.fv_vol_alpha_long;

                    let var_short_prev = v.local_vol_short * v.local_vol_short;
                    let var_long_prev = v.local_vol_long * v.local_vol_long;

                    let var_short_new = (1.0 - alpha_short) * var_short_prev + alpha_short * r2;
                    let var_long_new = (1.0 - alpha_long) * var_long_prev + alpha_long * r2;

                    v.local_vol_short = var_short_new.max(0.0).sqrt();
                    v.local_vol_long = var_long_new.max(0.0).sqrt();
                }
            }
        }
    }

    /// Run the engine's main tick.
    fn run_engine_tick(&mut self, now_ms: TimestampMs) {
        // Use inline engine logic to avoid lifetime issues
        // This duplicates Engine::main_tick but is necessary for ownership

        // 1) Update fair value and volatility
        self.update_fair_value_and_vol(now_ms);

        // 2) Mark-to-fair inventory / basis / unrealised PnL
        self.state.recompute_after_fills(&self.config);

        // 3) Toxicity + venue health
        crate::toxicity::update_toxicity_and_health(&mut self.state, &self.config, now_ms);

        // 4) Risk limits + regime / kill switch
        self.update_risk_limits_and_regime();
    }

    /// Update fair value and volatility (simplified inline version).
    fn update_fair_value_and_vol(&mut self, now_ms: TimestampMs) {
        // Collect venue mids for fair value estimation
        let mut mids: Vec<f64> = Vec::new();
        for v in &self.state.venues {
            if let Some(mid) = v.mid {
                if mid.is_finite() && mid > 0.0 {
                    mids.push(mid);
                }
            }
        }

        if mids.is_empty() {
            return;
        }

        // Simple median as fair value
        mids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = mids.len();
        let median = if n % 2 == 1 {
            mids[n / 2]
        } else {
            0.5 * (mids[n / 2 - 1] + mids[n / 2])
        };

        let prev_fair = self.state.fair_value.unwrap_or(self.state.fair_value_prev);
        self.state.fair_value_prev = prev_fair;
        self.state.fair_value = Some(median);
        self.state.fv_available = true;
        self.state.healthy_venues_used_count = mids.len();

        // Update volatility
        if prev_fair > 0.0 && median > 0.0 {
            let ret = (median / prev_fair).ln();
            let r2 = ret * ret;

            let alpha_short = self.config.volatility.fv_vol_alpha_short;
            let alpha_long = self.config.volatility.fv_vol_alpha_long;

            let var_short_prev = self.state.fv_short_vol * self.state.fv_short_vol;
            let var_long_prev = self.state.fv_long_vol * self.state.fv_long_vol;

            let var_short_new = (1.0 - alpha_short) * var_short_prev + alpha_short * r2;
            let var_long_new = (1.0 - alpha_long) * var_long_prev + alpha_long * r2;

            self.state.fv_short_vol = var_short_new.max(0.0).sqrt();
            self.state.fv_long_vol = var_long_new.max(0.0).sqrt();
        }

        // Effective vol with floor
        self.state.sigma_eff = self
            .state
            .fv_short_vol
            .max(self.config.volatility.sigma_min);

        // Vol ratio and scalars
        let vol_ratio = if self.config.volatility.vol_ref > 0.0 {
            self.state.sigma_eff / self.config.volatility.vol_ref
        } else {
            1.0
        };
        self.state.vol_ratio_clipped = vol_ratio.clamp(
            self.config.volatility.vol_ratio_min,
            self.config.volatility.vol_ratio_max,
        );

        let bump = (self.state.vol_ratio_clipped - 1.0).max(0.0);
        self.state.spread_mult = 1.0 + self.config.volatility.spread_vol_mult_coeff * bump;
        self.state.size_mult = 1.0 / (1.0 + self.config.volatility.size_vol_mult_coeff * bump);
        self.state.band_mult = 1.0 / (1.0 + self.config.volatility.band_vol_mult_coeff * bump);

        self.state.kf_last_update_ms = now_ms;
    }

    /// Update risk limits and regime (simplified inline version).
    fn update_risk_limits_and_regime(&mut self) {
        let risk_cfg = &self.config.risk;

        // Vol-scaled delta limit
        let vol_ratio = self.state.vol_ratio_clipped.max(1e-6);
        self.state.delta_limit_usd = risk_cfg.delta_hard_limit_usd_base / vol_ratio;

        self.state.basis_limit_hard_usd = risk_cfg.basis_hard_limit_usd;
        self.state.basis_limit_warn_usd = risk_cfg.basis_hard_limit_usd * risk_cfg.basis_warn_frac;

        // Aggregate daily PnL
        self.state.daily_pnl_total =
            self.state.daily_realised_pnl + self.state.daily_unrealised_pnl;

        // Hard breach conditions
        let delta_abs = self.state.dollar_delta_usd.abs();
        let basis_abs = self.state.basis_usd.abs();
        let pnl = self.state.daily_pnl_total;

        let loss_limit = -risk_cfg.daily_loss_limit.abs();

        let pnl_hard_breach = pnl <= loss_limit;
        let delta_hard_breach = delta_abs >= self.state.delta_limit_usd;
        let basis_hard_breach = basis_abs >= self.state.basis_limit_hard_usd;

        // Kill switch
        if !self.state.kill_switch {
            use crate::state::KillReason;
            if pnl_hard_breach {
                self.state.kill_switch = true;
                self.state.kill_reason = KillReason::PnlHardBreach;
            } else if delta_hard_breach {
                self.state.kill_switch = true;
                self.state.kill_reason = KillReason::DeltaHardBreach;
            } else if basis_hard_breach {
                self.state.kill_switch = true;
                self.state.kill_reason = KillReason::BasisHardBreach;
            }
        }

        // Risk regime
        use crate::state::RiskRegime;
        self.state.risk_regime = if self.state.kill_switch
            || pnl_hard_breach
            || delta_hard_breach
            || basis_hard_breach
        {
            RiskRegime::HardLimit
        } else {
            let delta_warn = risk_cfg.delta_warn_frac * self.state.delta_limit_usd;
            let basis_warn = self.state.basis_limit_warn_usd;
            let pnl_warn = loss_limit * risk_cfg.pnl_warn_frac;

            if delta_abs >= delta_warn || basis_abs >= basis_warn || pnl <= pnl_warn {
                RiskRegime::Warning
            } else {
                RiskRegime::Normal
            }
        };
    }

    /// Execute strategy with the given action.
    fn execute_strategy(&mut self, now_ms: TimestampMs, action: &PolicyAction) {
        if self.state.kill_switch {
            return;
        }

        let mut all_fills = Vec::new();

        // 1) Market-making
        let mm_quotes = compute_mm_quotes(&self.config, &self.state);
        let mm_intents = mm_quotes_to_order_intents(&mm_quotes);

        // Apply action modifiers to MM intents
        let modified_intents = self.apply_action_to_intents(&mm_intents, action);

        let mm_fills =
            self.gateway
                .process_intents(&self.config, &mut self.state, &modified_intents);

        self.record_markouts_for_fills(&mm_fills, now_ms);
        all_fills.extend(mm_fills);
        self.state.recompute_after_fills(&self.config);

        // 2) Exit engine
        if self.config.exit.enabled {
            let exit_intents = exit::compute_exit_intents(&self.config, &self.state, now_ms);
            if !exit_intents.is_empty() {
                let exit_fills =
                    self.gateway
                        .process_intents(&self.config, &mut self.state, &exit_intents);
                self.record_markouts_for_fills(&exit_fills, now_ms);
                all_fills.extend(exit_fills);
                self.state.recompute_after_fills(&self.config);
            }
        }

        // 3) Hedge engine
        let mut hedge_intents: Vec<OrderIntent> = Vec::new();
        if let Some(plan) = compute_hedge_plan(&self.config, &self.state, now_ms) {
            hedge_intents = hedge_plan_to_order_intents(&plan);
        }

        // Apply hedge scale from action
        let hedge_intents = self.apply_hedge_action(&hedge_intents, action);

        if !hedge_intents.is_empty() {
            let hedge_fills =
                self.gateway
                    .process_intents(&self.config, &mut self.state, &hedge_intents);
            self.record_markouts_for_fills(&hedge_fills, now_ms);
            all_fills.extend(hedge_fills);
            self.state.recompute_after_fills(&self.config);
        }
    }

    /// Apply action modifiers to MM intents.
    fn apply_action_to_intents(
        &self,
        intents: &[OrderIntent],
        action: &PolicyAction,
    ) -> Vec<OrderIntent> {
        intents
            .iter()
            .map(|intent| {
                let mut modified = intent.clone();

                // Apply size scale
                if intent.venue_index < action.size_scale.len() {
                    modified.size *= action.size_scale[intent.venue_index];
                }

                // Apply spread scale to price (simplified)
                if intent.venue_index < action.spread_scale.len() {
                    let scale = action.spread_scale[intent.venue_index];
                    let fair = self.state.fair_value.unwrap_or(intent.price);
                    let offset = intent.price - fair;
                    modified.price = fair + offset * scale;
                }

                modified
            })
            .collect()
    }

    /// Apply hedge action modifiers.
    fn apply_hedge_action(
        &self,
        intents: &[OrderIntent],
        action: &PolicyAction,
    ) -> Vec<OrderIntent> {
        intents
            .iter()
            .map(|intent| {
                let mut modified = intent.clone();
                modified.size *= action.hedge_scale;
                modified
            })
            .collect()
    }

    /// Record pending markouts for fills.
    fn record_markouts_for_fills(&mut self, fills: &[FillEvent], now_ms: TimestampMs) {
        let tox_cfg = &self.config.toxicity;
        let horizon_ms = tox_cfg.markout_horizon_ms;
        let max_pending = tox_cfg.max_pending_per_venue;

        let fair = self
            .state
            .fair_value
            .unwrap_or(self.state.fair_value_prev)
            .max(1.0);

        for fill in fills {
            let mid = self
                .state
                .venues
                .get(fill.venue_index)
                .and_then(|v| v.mid)
                .unwrap_or(fair);

            self.state.record_pending_markout(PendingMarkoutRecord {
                venue_index: fill.venue_index,
                side: fill.side,
                size_tao: fill.size,
                price: fill.price,
                now_ms,
                fair,
                mid,
                horizon_ms,
                max_pending,
            });
        }
    }

    /// Check termination conditions.
    fn check_termination(&self) -> Option<String> {
        if self.state.kill_switch {
            return Some(format!("KillSwitch: {:?}", self.state.kill_reason));
        }
        if self.tick >= self.env_config.max_ticks {
            return Some("MaxTicks".to_string());
        }
        None
    }

    /// Compute reward for this step.
    fn compute_reward(&self) -> (f64, RewardComponents) {
        let obs = Observation::from_state(
            &self.state,
            &self.config,
            self.base_ms + (self.tick as i64) * self.env_config.dt_ms,
            self.tick,
        );

        let components = RewardComponents::from_observation(&obs, self.prev_pnl, self.peak_pnl);
        let reward = components.compute_reward(&self.env_config.reward_weights);

        (reward, components)
    }

    /// Build step info dictionary.
    fn build_step_info(&self, termination_reason: Option<String>) -> StepInfo {
        StepInfo {
            termination_reason,
            tick: self.tick,
            pnl_total: self.state.daily_pnl_total,
            pnl_realised: self.state.daily_realised_pnl,
            pnl_unrealised: self.state.daily_unrealised_pnl,
            kill_switch: self.state.kill_switch,
            kill_reason: if self.state.kill_switch {
                Some(format!("{:?}", self.state.kill_reason))
            } else {
                None
            },
            risk_regime: format!("{:?}", self.state.risk_regime),
            q_global_tao: self.state.q_global_tao,
            dollar_delta_usd: self.state.dollar_delta_usd,
            basis_usd: self.state.basis_usd,
            domain_rand_sample: self.current_domain_rand.clone(),
            reward_components: None,
        }
    }
}

/// Vectorised environment for parallel rollouts.
///
/// Manages N independent SimEnv instances.
pub struct VecEnv {
    /// Individual environments.
    envs: Vec<SimEnv>,
}

impl VecEnv {
    /// Create a new vectorised environment with N copies.
    pub fn new(n: usize, base_config: Config, env_config: SimEnvConfig) -> Self {
        let envs = (0..n)
            .map(|_| SimEnv::new(base_config.clone(), env_config.clone()))
            .collect();
        Self { envs }
    }

    /// Get the number of environments.
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Get the number of venues per environment.
    pub fn num_venues(&self) -> usize {
        self.envs.first().map(|e| e.num_venues()).unwrap_or(0)
    }

    /// Reset all environments with optional per-environment seeds.
    ///
    /// If seeds is None, random seeds are generated.
    /// If seeds has fewer elements than envs, remaining envs get random seeds.
    pub fn reset_all(&mut self, seeds: Option<&[u64]>) -> Vec<Observation> {
        self.envs
            .iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let seed = seeds.and_then(|s| s.get(i).copied());
                env.reset(seed)
            })
            .collect()
    }

    /// Step all environments with the given actions.
    ///
    /// Actions must have the same length as envs.
    pub fn step(&mut self, actions: &[PolicyAction]) -> Vec<StepResult> {
        assert_eq!(
            actions.len(),
            self.envs.len(),
            "Actions length must match number of environments"
        );

        self.envs
            .iter_mut()
            .zip(actions.iter())
            .map(|(env, action)| env.step(action))
            .collect()
    }

    /// Step all environments with identity actions (baseline behavior).
    pub fn step_identity(&mut self) -> Vec<StepResult> {
        let actions: Vec<_> = self
            .envs
            .iter()
            .map(|env| PolicyAction::identity(env.num_venues(), "identity"))
            .collect();
        self.step(&actions)
    }

    /// Get all environment states (for testing).
    pub fn states(&self) -> Vec<&GlobalState> {
        self.envs.iter().map(|e| e.state()).collect()
    }

    /// Get all current seeds.
    pub fn seeds(&self) -> Vec<u64> {
        self.envs.iter().map(|e| e.seed()).collect()
    }

    /// Check which environments are done.
    pub fn dones(&self) -> Vec<bool> {
        self.envs.iter().map(|e| e.is_done()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_sim_env_reset() {
        let config = make_default_config();
        let env_config = SimEnvConfig::deterministic();
        let mut env = SimEnv::new(config, env_config);

        let obs = env.reset(Some(42));

        assert_eq!(obs.tick_index, 0);
        assert!(obs.fair_value.is_some());
        assert!(!env.is_done());
    }

    #[test]
    fn test_sim_env_step() {
        let config = make_default_config();
        let env_config = SimEnvConfig::deterministic();
        let mut env = SimEnv::new(config, env_config);

        env.reset(Some(42));

        let action = PolicyAction::identity(env.num_venues(), "test");
        let result = env.step(&action);

        assert!(!result.done);
        assert_eq!(result.info.tick, 1);
    }

    #[test]
    fn test_sim_env_determinism() {
        let config = make_default_config();
        let env_config = SimEnvConfig::deterministic();

        // Run 1
        let mut env1 = SimEnv::new(config.clone(), env_config.clone());
        let obs1 = env1.reset(Some(42));
        let action = PolicyAction::identity(env1.num_venues(), "test");
        let results1: Vec<_> = (0..10).map(|_| env1.step(&action)).collect();

        // Run 2 with same seed
        let mut env2 = SimEnv::new(config, env_config);
        let obs2 = env2.reset(Some(42));
        let results2: Vec<_> = (0..10).map(|_| env2.step(&action)).collect();

        // Should be identical
        assert_eq!(obs1.fair_value, obs2.fair_value);
        assert_eq!(obs1.q_global_tao, obs2.q_global_tao);

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.observation.fair_value, r2.observation.fair_value);
            assert_eq!(r1.observation.q_global_tao, r2.observation.q_global_tao);
            assert!((r1.reward - r2.reward).abs() < 1e-9);
            assert_eq!(r1.done, r2.done);
        }
    }

    #[test]
    fn test_sim_env_different_seeds_different_results() {
        let config = make_default_config();
        let env_config = SimEnvConfig::default(); // With randomisation

        let mut env1 = SimEnv::new(config.clone(), env_config.clone());
        let obs1 = env1.reset(Some(42));

        let mut env2 = SimEnv::new(config, env_config);
        let obs2 = env2.reset(Some(43));

        // With domain randomisation, different seeds should produce different states
        // (though fair_value might be the same if not affected by rand)
        assert!(
            obs1.q_global_tao != obs2.q_global_tao
                || obs1.fair_value != obs2.fair_value
                || obs1.sigma_eff != obs2.sigma_eff
        );
    }

    #[test]
    fn test_vec_env_basic() {
        let config = make_default_config();
        let env_config = SimEnvConfig::deterministic();

        let mut vec_env = VecEnv::new(4, config, env_config);

        assert_eq!(vec_env.num_envs(), 4);

        // Reset all
        let seeds = vec![10, 20, 30, 40];
        let observations = vec_env.reset_all(Some(&seeds));

        assert_eq!(observations.len(), 4);

        // Step all with identity actions
        let results = vec_env.step_identity();

        assert_eq!(results.len(), 4);
        for result in &results {
            assert!(!result.done);
        }
    }

    #[test]
    fn test_vec_env_determinism() {
        let config = make_default_config();
        let env_config = SimEnvConfig::deterministic();

        let seeds = vec![100, 200, 300, 400];

        // Run 1
        let mut vec_env1 = VecEnv::new(4, config.clone(), env_config.clone());
        let obs1 = vec_env1.reset_all(Some(&seeds));
        let results1: Vec<Vec<_>> = (0..5).map(|_| vec_env1.step_identity()).collect();

        // Run 2 with same seeds
        let mut vec_env2 = VecEnv::new(4, config, env_config);
        let obs2 = vec_env2.reset_all(Some(&seeds));
        let results2: Vec<Vec<_>> = (0..5).map(|_| vec_env2.step_identity()).collect();

        // Compare observations
        for (o1, o2) in obs1.iter().zip(obs2.iter()) {
            assert_eq!(o1.fair_value, o2.fair_value);
            assert_eq!(o1.q_global_tao, o2.q_global_tao);
        }

        // Compare step results
        for (r1_batch, r2_batch) in results1.iter().zip(results2.iter()) {
            for (r1, r2) in r1_batch.iter().zip(r2_batch.iter()) {
                assert!((r1.reward - r2.reward).abs() < 1e-9);
                assert_eq!(r1.done, r2.done);
            }
        }
    }

    #[test]
    fn test_episode_terminates_on_max_ticks() {
        let config = make_default_config();
        let mut env_config = SimEnvConfig::deterministic();
        env_config.max_ticks = 10;

        let mut env = SimEnv::new(config, env_config);
        env.reset(Some(42));

        let action = PolicyAction::identity(env.num_venues(), "test");

        for _ in 0..9 {
            let result = env.step(&action);
            assert!(!result.done);
        }

        let result = env.step(&action);
        assert!(result.done);
        assert_eq!(result.info.termination_reason, Some("MaxTicks".to_string()));
    }
}
