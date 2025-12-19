// src/rl/domain_rand.rs
//
// Domain randomisation for RL-1 training environments.
//
// Per ROADMAP.md RL-1, this module provides:
// - Configuration struct for randomisation ranges
// - Deterministic sampler using seeded RNG
// - Support for: fees, spreads, slippage, funding, volatility, venue events
//
// All sampling is deterministic given a seed.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Configuration for domain randomisation ranges.
///
/// Each field specifies a (min, max) range. The sampler will uniformly
/// sample within these ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRandConfig {
    // ----- Fee randomisation -----
    /// Maker fee range in basis points per venue.
    pub maker_fee_bps_range: (f64, f64),
    /// Taker fee range in basis points per venue.
    pub taker_fee_bps_range: (f64, f64),
    /// Maker rebate range in basis points per venue.
    pub maker_rebate_bps_range: (f64, f64),

    // ----- Spread randomisation -----
    /// Base spread multiplier range (applied to all venues).
    pub spread_mult_range: (f64, f64),
    /// Per-venue spread offset range (additive, in USD).
    pub spread_offset_range: (f64, f64),

    // ----- Slippage model randomisation -----
    /// Linear slippage coefficient range (USD per TAO).
    pub slippage_linear_range: (f64, f64),
    /// Quadratic slippage coefficient range (USD per TAO^2).
    pub slippage_quadratic_range: (f64, f64),

    // ----- Funding regime randomisation -----
    /// 8h funding rate range (dimensionless, e.g. -0.01 to 0.01).
    pub funding_8h_range: (f64, f64),
    /// Probability of "funding spike" event per episode.
    pub funding_spike_prob: f64,
    /// Magnitude of funding spike when it occurs.
    pub funding_spike_magnitude: f64,

    // ----- Volatility regime randomisation -----
    /// Reference volatility range (sigma_ref).
    pub vol_ref_range: (f64, f64),
    /// Short-term vol alpha range.
    pub vol_alpha_short_range: (f64, f64),
    /// Long-term vol alpha range.
    pub vol_alpha_long_range: (f64, f64),
    /// Probability of "high volatility regime" per episode.
    pub high_vol_regime_prob: f64,
    /// Volatility multiplier during high-vol regime.
    pub high_vol_multiplier: f64,

    // ----- Venue staleness / disable events -----
    /// Probability of a venue becoming stale per tick.
    pub venue_stale_prob_per_tick: f64,
    /// Duration of staleness in ticks.
    pub venue_stale_duration_ticks: (u64, u64),
    /// Probability of venue disable event per episode.
    pub venue_disable_prob: f64,
    /// Duration of disable event in ticks.
    pub venue_disable_duration_ticks: (u64, u64),

    // ----- Depth randomisation -----
    /// Depth near mid range (USD).
    pub depth_near_mid_range: (f64, f64),
    /// Probability of "thin book" event per tick.
    pub thin_book_prob_per_tick: f64,
    /// Depth multiplier during thin book event.
    pub thin_book_depth_mult: f64,

    // ----- Initial conditions -----
    /// Initial inventory range (TAO).
    pub initial_q_tao_range: (f64, f64),
    /// Initial fair value range (USD).
    pub initial_fair_value_range: (f64, f64),
}

impl Default for DomainRandConfig {
    fn default() -> Self {
        Self {
            // Fee ranges (typical exchange ranges)
            maker_fee_bps_range: (0.5, 3.0),
            taker_fee_bps_range: (2.0, 7.0),
            maker_rebate_bps_range: (0.0, 1.0),

            // Spread randomisation
            spread_mult_range: (0.8, 1.5),
            spread_offset_range: (-0.1, 0.1),

            // Slippage model
            slippage_linear_range: (0.005, 0.02),
            slippage_quadratic_range: (0.0005, 0.002),

            // Funding regime
            funding_8h_range: (-0.005, 0.005),
            funding_spike_prob: 0.05,
            funding_spike_magnitude: 0.02,

            // Volatility regime
            vol_ref_range: (0.015, 0.045),
            vol_alpha_short_range: (0.1, 0.3),
            vol_alpha_long_range: (0.03, 0.08),
            high_vol_regime_prob: 0.1,
            high_vol_multiplier: 2.0,

            // Venue events
            venue_stale_prob_per_tick: 0.001,
            venue_stale_duration_ticks: (5, 20),
            venue_disable_prob: 0.02,
            venue_disable_duration_ticks: (50, 200),

            // Depth
            depth_near_mid_range: (5000.0, 20000.0),
            thin_book_prob_per_tick: 0.005,
            thin_book_depth_mult: 0.2,

            // Initial conditions
            initial_q_tao_range: (-10.0, 10.0),
            initial_fair_value_range: (200.0, 400.0),
        }
    }
}

impl DomainRandConfig {
    /// Create a config with no randomisation (all ranges collapsed to defaults).
    pub fn deterministic() -> Self {
        Self {
            maker_fee_bps_range: (2.0, 2.0),
            taker_fee_bps_range: (5.0, 5.0),
            maker_rebate_bps_range: (0.0, 0.0),
            spread_mult_range: (1.0, 1.0),
            spread_offset_range: (0.0, 0.0),
            slippage_linear_range: (0.01, 0.01),
            slippage_quadratic_range: (0.001, 0.001),
            funding_8h_range: (0.0, 0.0),
            funding_spike_prob: 0.0,
            funding_spike_magnitude: 0.0,
            vol_ref_range: (0.028, 0.028),
            vol_alpha_short_range: (0.2, 0.2),
            vol_alpha_long_range: (0.05, 0.05),
            high_vol_regime_prob: 0.0,
            high_vol_multiplier: 1.0,
            venue_stale_prob_per_tick: 0.0,
            venue_stale_duration_ticks: (0, 0),
            venue_disable_prob: 0.0,
            venue_disable_duration_ticks: (0, 0),
            depth_near_mid_range: (10000.0, 10000.0),
            thin_book_prob_per_tick: 0.0,
            thin_book_depth_mult: 1.0,
            initial_q_tao_range: (0.0, 0.0),
            initial_fair_value_range: (300.0, 300.0),
        }
    }

    /// Create a config with mild randomisation (for stable training).
    pub fn mild() -> Self {
        Self {
            maker_fee_bps_range: (1.5, 2.5),
            taker_fee_bps_range: (4.0, 6.0),
            maker_rebate_bps_range: (0.0, 0.5),
            spread_mult_range: (0.9, 1.2),
            spread_offset_range: (-0.05, 0.05),
            slippage_linear_range: (0.008, 0.015),
            slippage_quadratic_range: (0.0008, 0.0015),
            funding_8h_range: (-0.002, 0.002),
            funding_spike_prob: 0.02,
            funding_spike_magnitude: 0.01,
            vol_ref_range: (0.02, 0.035),
            vol_alpha_short_range: (0.15, 0.25),
            vol_alpha_long_range: (0.04, 0.06),
            high_vol_regime_prob: 0.05,
            high_vol_multiplier: 1.5,
            venue_stale_prob_per_tick: 0.0005,
            venue_stale_duration_ticks: (3, 10),
            venue_disable_prob: 0.01,
            venue_disable_duration_ticks: (20, 100),
            depth_near_mid_range: (8000.0, 15000.0),
            thin_book_prob_per_tick: 0.002,
            thin_book_depth_mult: 0.4,
            initial_q_tao_range: (-5.0, 5.0),
            initial_fair_value_range: (250.0, 350.0),
        }
    }
}

/// Sampled domain randomisation parameters for a single episode.
///
/// These are the concrete values sampled from `DomainRandConfig` ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRandSample {
    /// Per-venue maker fees (bps).
    pub maker_fees_bps: Vec<f64>,
    /// Per-venue taker fees (bps).
    pub taker_fees_bps: Vec<f64>,
    /// Per-venue maker rebates (bps).
    pub maker_rebates_bps: Vec<f64>,
    /// Global spread multiplier.
    pub spread_mult: f64,
    /// Per-venue spread offsets (USD).
    pub spread_offsets: Vec<f64>,
    /// Linear slippage coefficient.
    pub slippage_linear: f64,
    /// Quadratic slippage coefficient.
    pub slippage_quadratic: f64,
    /// Per-venue 8h funding rates.
    pub funding_8h: Vec<f64>,
    /// Whether a funding spike occurs this episode.
    pub funding_spike_active: bool,
    /// Reference volatility.
    pub vol_ref: f64,
    /// Short-term vol alpha.
    pub vol_alpha_short: f64,
    /// Long-term vol alpha.
    pub vol_alpha_long: f64,
    /// Whether high-vol regime is active this episode.
    pub high_vol_regime: bool,
    /// Volatility multiplier (1.0 if normal, higher if high_vol_regime).
    pub vol_multiplier: f64,
    /// Per-venue base depth near mid.
    pub depth_near_mid: Vec<f64>,
    /// Initial inventory (TAO).
    pub initial_q_tao: f64,
    /// Initial fair value (USD).
    pub initial_fair_value: f64,
}

/// Deterministic domain randomisation sampler.
///
/// Given a seed, produces reproducible randomisation samples.
pub struct DomainRandSampler {
    config: DomainRandConfig,
    rng: ChaCha8Rng,
}

impl DomainRandSampler {
    /// Create a new sampler with the given config and seed.
    pub fn new(config: DomainRandConfig, seed: u64) -> Self {
        Self {
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Reseed the RNG.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Sample a uniform value in the given range.
    fn sample_range(&mut self, range: (f64, f64)) -> f64 {
        if range.0 >= range.1 {
            return range.0;
        }
        self.rng.gen_range(range.0..=range.1)
    }

    /// Sample a uniform u64 in the given range.
    fn sample_range_u64(&mut self, range: (u64, u64)) -> u64 {
        if range.0 >= range.1 {
            return range.0;
        }
        self.rng.gen_range(range.0..=range.1)
    }

    /// Sample a boolean with given probability.
    fn sample_bool(&mut self, prob: f64) -> bool {
        self.rng.gen::<f64>() < prob
    }

    /// Sample domain randomisation parameters for a single episode.
    pub fn sample_episode(&mut self, num_venues: usize) -> DomainRandSample {
        // Copy ranges to avoid borrowing conflicts with &mut self in closures
        let maker_fee_range = self.config.maker_fee_bps_range;
        let taker_fee_range = self.config.taker_fee_bps_range;
        let maker_rebate_range = self.config.maker_rebate_bps_range;
        let spread_mult_range = self.config.spread_mult_range;
        let spread_offset_range = self.config.spread_offset_range;
        let slippage_linear_range = self.config.slippage_linear_range;
        let slippage_quadratic_range = self.config.slippage_quadratic_range;
        let funding_8h_range = self.config.funding_8h_range;
        let funding_spike_prob = self.config.funding_spike_prob;
        let funding_spike_magnitude = self.config.funding_spike_magnitude;
        let vol_ref_range = self.config.vol_ref_range;
        let vol_alpha_short_range = self.config.vol_alpha_short_range;
        let vol_alpha_long_range = self.config.vol_alpha_long_range;
        let high_vol_regime_prob = self.config.high_vol_regime_prob;
        let high_vol_multiplier = self.config.high_vol_multiplier;
        let depth_near_mid_range = self.config.depth_near_mid_range;
        let initial_q_tao_range = self.config.initial_q_tao_range;
        let initial_fair_value_range = self.config.initial_fair_value_range;

        // Per-venue fees
        let mut maker_fees_bps = Vec::with_capacity(num_venues);
        let mut taker_fees_bps = Vec::with_capacity(num_venues);
        let mut maker_rebates_bps = Vec::with_capacity(num_venues);
        for _ in 0..num_venues {
            maker_fees_bps.push(self.sample_range(maker_fee_range));
            taker_fees_bps.push(self.sample_range(taker_fee_range));
            maker_rebates_bps.push(self.sample_range(maker_rebate_range));
        }

        // Spreads
        let spread_mult = self.sample_range(spread_mult_range);
        let mut spread_offsets = Vec::with_capacity(num_venues);
        for _ in 0..num_venues {
            spread_offsets.push(self.sample_range(spread_offset_range));
        }

        // Slippage
        let slippage_linear = self.sample_range(slippage_linear_range);
        let slippage_quadratic = self.sample_range(slippage_quadratic_range);

        // Funding
        let mut base_funding = Vec::with_capacity(num_venues);
        for _ in 0..num_venues {
            base_funding.push(self.sample_range(funding_8h_range));
        }
        let funding_spike_active = self.sample_bool(funding_spike_prob);
        let funding_8h: Vec<f64> = if funding_spike_active {
            base_funding
                .iter()
                .map(|f| f + funding_spike_magnitude * self.rng.gen_range(-1.0..=1.0))
                .collect()
        } else {
            base_funding
        };

        // Volatility
        let vol_ref = self.sample_range(vol_ref_range);
        let vol_alpha_short = self.sample_range(vol_alpha_short_range);
        let vol_alpha_long = self.sample_range(vol_alpha_long_range);
        let high_vol_regime = self.sample_bool(high_vol_regime_prob);
        let vol_multiplier = if high_vol_regime {
            high_vol_multiplier
        } else {
            1.0
        };

        // Depth
        let mut depth_near_mid = Vec::with_capacity(num_venues);
        for _ in 0..num_venues {
            depth_near_mid.push(self.sample_range(depth_near_mid_range));
        }

        // Initial conditions
        let initial_q_tao = self.sample_range(initial_q_tao_range);
        let initial_fair_value = self.sample_range(initial_fair_value_range);

        DomainRandSample {
            maker_fees_bps,
            taker_fees_bps,
            maker_rebates_bps,
            spread_mult,
            spread_offsets,
            slippage_linear,
            slippage_quadratic,
            funding_8h,
            funding_spike_active,
            vol_ref,
            vol_alpha_short,
            vol_alpha_long,
            high_vol_regime,
            vol_multiplier,
            depth_near_mid,
            initial_q_tao,
            initial_fair_value,
        }
    }

    /// Sample venue event for a single tick.
    ///
    /// Returns (is_stale, stale_duration, is_disabled, disable_duration) for each venue.
    pub fn sample_venue_events(&mut self, num_venues: usize) -> Vec<(bool, u64, bool, u64)> {
        // Copy config values to avoid borrowing conflicts
        let venue_stale_prob = self.config.venue_stale_prob_per_tick;
        let venue_stale_duration = self.config.venue_stale_duration_ticks;
        let venue_disable_prob = self.config.venue_disable_prob;
        let venue_disable_duration = self.config.venue_disable_duration_ticks;

        let mut results = Vec::with_capacity(num_venues);
        for _ in 0..num_venues {
            let is_stale = self.sample_bool(venue_stale_prob);
            let stale_duration = if is_stale {
                self.sample_range_u64(venue_stale_duration)
            } else {
                0
            };
            // Disable events are per-episode, but we sample per-tick for flexibility
            let is_disabled = self.sample_bool(venue_disable_prob / 1000.0); // Scale down
            let disable_duration = if is_disabled {
                self.sample_range_u64(venue_disable_duration)
            } else {
                0
            };
            results.push((is_stale, stale_duration, is_disabled, disable_duration));
        }
        results
    }

    /// Sample thin book event.
    ///
    /// Returns true if thin book condition should be applied this tick.
    pub fn sample_thin_book(&mut self) -> bool {
        self.sample_bool(self.config.thin_book_prob_per_tick)
    }

    /// Get the thin book depth multiplier.
    pub fn thin_book_depth_mult(&self) -> f64 {
        self.config.thin_book_depth_mult
    }

    /// Get reference to config.
    pub fn config(&self) -> &DomainRandConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_rand_determinism() {
        let config = DomainRandConfig::default();

        // Sample with same seed twice
        let mut sampler1 = DomainRandSampler::new(config.clone(), 42);
        let sample1 = sampler1.sample_episode(5);

        let mut sampler2 = DomainRandSampler::new(config, 42);
        let sample2 = sampler2.sample_episode(5);

        // Should be identical
        assert_eq!(sample1.initial_q_tao, sample2.initial_q_tao);
        assert_eq!(sample1.initial_fair_value, sample2.initial_fair_value);
        assert_eq!(sample1.vol_ref, sample2.vol_ref);
        assert_eq!(sample1.maker_fees_bps, sample2.maker_fees_bps);
        assert_eq!(sample1.spread_mult, sample2.spread_mult);
    }

    #[test]
    fn test_domain_rand_different_seeds() {
        let config = DomainRandConfig::default();

        let mut sampler1 = DomainRandSampler::new(config.clone(), 42);
        let sample1 = sampler1.sample_episode(5);

        let mut sampler2 = DomainRandSampler::new(config, 43);
        let sample2 = sampler2.sample_episode(5);

        // Should be different (with very high probability)
        assert!(
            sample1.initial_q_tao != sample2.initial_q_tao
                || sample1.initial_fair_value != sample2.initial_fair_value
        );
    }

    #[test]
    fn test_deterministic_config() {
        let config = DomainRandConfig::deterministic();
        let mut sampler = DomainRandSampler::new(config, 42);
        let sample = sampler.sample_episode(5);

        // All values should be at the fixed point
        assert_eq!(sample.initial_q_tao, 0.0);
        assert_eq!(sample.initial_fair_value, 300.0);
        assert_eq!(sample.spread_mult, 1.0);
        assert!(!sample.funding_spike_active);
        assert!(!sample.high_vol_regime);
    }

    #[test]
    fn test_sample_ranges() {
        let config = DomainRandConfig::default();
        let mut sampler = DomainRandSampler::new(config.clone(), 12345);

        for _ in 0..100 {
            let sample = sampler.sample_episode(5);

            // Check all values are within ranges
            assert!(sample.initial_q_tao >= config.initial_q_tao_range.0);
            assert!(sample.initial_q_tao <= config.initial_q_tao_range.1);
            assert!(sample.initial_fair_value >= config.initial_fair_value_range.0);
            assert!(sample.initial_fair_value <= config.initial_fair_value_range.1);
            assert!(sample.vol_ref >= config.vol_ref_range.0);
            assert!(sample.vol_ref <= config.vol_ref_range.1);
        }
    }
}
