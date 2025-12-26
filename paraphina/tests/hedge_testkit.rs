// tests/hedge_testkit.rs
//
// Shared test utilities for hedge allocator verification tests.
// This module provides common setup helpers and a deterministic PRNG
// for property-style invariant testing.
//
// Note: This module is included via #[path] from other test files.
// The dead_code warnings are suppressed because not all functions
// are used in every test file that includes this module.

#![allow(dead_code)]

use paraphina::config::Config;
use paraphina::state::GlobalState;
use paraphina::types::VenueStatus;

// ============================================================================
// DETERMINISTIC PRNG (xorshift64)
// ============================================================================

/// A minimal deterministic PRNG using xorshift64.
/// This is NOT cryptographically secure, but it is fast, deterministic,
/// and sufficient for generating test inputs.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG with the given seed.
    /// Seed must be non-zero.
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "Seed must be non-zero");
        Xorshift64 { state: seed }
    }

    /// Generate the next u64 value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate a f64 in [lo, hi).
    pub fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Generate a usize in [0, n).
    pub fn range_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    /// Generate a bool with probability p.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Pick one element from a slice.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.range_usize(items.len())]
    }
}

// ============================================================================
// TEST CASE GENERATOR
// ============================================================================

/// Parameters for a single test case.
#[derive(Debug, Clone)]
pub struct HedgeTestCase {
    /// Test case identifier for debugging.
    pub case_id: u64,
    /// Global inventory in TAO (can be positive or negative).
    pub q_global_tao: f64,
    /// Fair value price in USD.
    pub fair_value: f64,
    /// Volatility ratio (1.0 = normal).
    pub vol_ratio: f64,
    /// Configuration overrides.
    pub max_step_tao: f64,
    pub max_venue_tao_per_tick: f64,
    pub band_base_tao: f64,
    pub k_hedge: f64,
    pub margin_safety_buffer: f64,
    pub max_leverage: f64,
    pub chunk_size_tao: f64,
    pub chunk_convexity_cost_bps: f64,
    /// Per-venue configurations.
    pub venues: Vec<VenueTestConfig>,
}

/// Per-venue test configuration.
#[derive(Debug, Clone)]
pub struct VenueTestConfig {
    pub mid: f64,
    pub spread: f64,
    pub depth_usd: f64,
    pub position_tao: f64,
    pub margin_available_usd: f64,
    pub dist_liq_sigma: f64,
    pub funding_8h: f64,
    pub taker_fee_bps: f64,
    pub is_healthy: bool,
    pub is_stale: bool,
}

impl HedgeTestCase {
    /// Generate a random test case using the PRNG.
    pub fn random(rng: &mut Xorshift64, case_id: u64) -> Self {
        // Global parameters
        let q_sign = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
        let q_global_tao = q_sign * rng.range_f64(0.0, 100.0);
        let fair_value = rng.range_f64(50.0, 200.0);
        let vol_ratio = rng.range_f64(0.5, 2.0);

        // Configuration parameters
        let max_step_tao = rng.range_f64(1.0, 50.0);
        let max_venue_tao_per_tick = rng.range_f64(1.0, 30.0);
        let band_base_tao = rng.range_f64(0.5, 10.0);
        let k_hedge = rng.range_f64(0.1, 1.0);
        let margin_safety_buffer = rng.range_f64(0.8, 1.0);
        let max_leverage = *rng.pick(&[1.0, 5.0, 10.0, 20.0]);
        let chunk_size_tao = rng.range_f64(0.5, 10.0);
        let chunk_convexity_cost_bps = if rng.bernoulli(0.3) {
            rng.range_f64(0.0, 100.0)
        } else {
            0.0
        };

        // Generate 2-5 venues
        let num_venues = 2 + rng.range_usize(4);
        let venues: Vec<VenueTestConfig> = (0..num_venues)
            .map(|_| VenueTestConfig::random(rng, fair_value))
            .collect();

        HedgeTestCase {
            case_id,
            q_global_tao,
            fair_value,
            vol_ratio,
            max_step_tao,
            max_venue_tao_per_tick,
            band_base_tao,
            k_hedge,
            margin_safety_buffer,
            max_leverage,
            chunk_size_tao,
            chunk_convexity_cost_bps,
            venues,
        }
    }

    /// Build a Config from this test case.
    pub fn build_config(&self) -> Config {
        let mut cfg = Config::default();

        cfg.hedge.max_step_tao = self.max_step_tao;
        cfg.hedge.max_venue_tao_per_tick = self.max_venue_tao_per_tick;
        cfg.hedge.band_base_tao = self.band_base_tao;
        cfg.hedge.band_vol_mult = 0.5; // Fixed for simplicity
        cfg.hedge.k_hedge = self.k_hedge;
        cfg.hedge.margin_safety_buffer = self.margin_safety_buffer;
        cfg.hedge.max_leverage = self.max_leverage;
        cfg.hedge.chunk_size_tao = self.chunk_size_tao;
        cfg.hedge.chunk_convexity_cost_bps = self.chunk_convexity_cost_bps;
        cfg.hedge.min_depth_usd = 100.0;
        cfg.hedge.depth_fraction = 0.5;

        // Set venue fees
        for (i, v) in self.venues.iter().enumerate() {
            if i < cfg.venues.len() {
                cfg.venues[i].taker_fee_bps = v.taker_fee_bps;
            }
        }

        cfg
    }

    /// Build a GlobalState from this test case.
    ///
    /// Note: q_global_tao is derived from venue positions after recompute_after_fills.
    /// To get the desired q_global_tao, we put the remaining position on venue 0.
    pub fn build_state(&self, cfg: &Config) -> GlobalState {
        let mut state = GlobalState::new(cfg);

        state.fair_value = Some(self.fair_value);
        state.fair_value_prev = self.fair_value;
        state.vol_ratio_clipped = self.vol_ratio;

        for (i, v) in self.venues.iter().enumerate() {
            if i >= state.venues.len() {
                break;
            }

            state.venues[i].mid = Some(v.mid);
            state.venues[i].spread = Some(v.spread);
            state.venues[i].depth_near_mid = v.depth_usd;
            state.venues[i].position_tao = v.position_tao;
            state.venues[i].margin_available_usd = v.margin_available_usd;
            state.venues[i].dist_liq_sigma = v.dist_liq_sigma;
            state.venues[i].funding_8h = v.funding_8h;

            if v.is_healthy {
                state.venues[i].status = VenueStatus::Healthy;
                state.venues[i].last_mid_update_ms = Some(0);
            } else if v.is_stale {
                state.venues[i].status = VenueStatus::Healthy;
                state.venues[i].last_mid_update_ms = Some(-10000); // Very old
            } else {
                state.venues[i].status = VenueStatus::Disabled;
            }
        }

        // Calculate current sum of venue positions
        let venue_position_sum: f64 = self.venues.iter().map(|v| v.position_tao).sum();

        // If venue positions don't sum to q_global_tao, add the difference to venue 0
        // This ensures q_global_tao matches expectations after recompute_after_fills
        if (venue_position_sum - self.q_global_tao).abs() > 1e-9 && !self.venues.is_empty() {
            state.venues[0].position_tao = self.q_global_tao
                - self
                    .venues
                    .iter()
                    .skip(1)
                    .map(|v| v.position_tao)
                    .sum::<f64>();
        }

        state.recompute_after_fills(cfg);
        state
    }
}

impl VenueTestConfig {
    /// Generate a random venue configuration.
    pub fn random(rng: &mut Xorshift64, fair_value: f64) -> Self {
        // Mid price slightly deviates from fair
        let mid = fair_value * rng.range_f64(0.98, 1.02);
        let spread = mid * rng.range_f64(0.001, 0.01);
        let depth_usd = rng.range_f64(50.0, 500_000.0);

        // Position can be positive, negative, or zero
        let position_tao = if rng.bernoulli(0.3) {
            0.0
        } else {
            let sign = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
            sign * rng.range_f64(0.0, 50.0)
        };

        let margin_available_usd = rng.range_f64(0.0, 50_000.0);
        let dist_liq_sigma = rng.range_f64(0.5, 20.0);
        let funding_8h = rng.range_f64(-0.01, 0.01);
        let taker_fee_bps = rng.range_f64(1.0, 20.0);

        // Most venues healthy, some disabled/stale
        let is_healthy = rng.bernoulli(0.8);
        let is_stale = !is_healthy && rng.bernoulli(0.5);

        VenueTestConfig {
            mid,
            spread,
            depth_usd,
            position_tao,
            margin_available_usd,
            dist_liq_sigma,
            funding_8h,
            taker_fee_bps,
            is_healthy,
            is_stale,
        }
    }
}

// ============================================================================
// SHARED TEST HELPERS
// ============================================================================

/// Set up a venue with standard book data.
pub fn setup_venue_book(
    state: &mut GlobalState,
    index: usize,
    mid: f64,
    spread: f64,
    depth: f64,
    update_time: i64,
) {
    state.venues[index].mid = Some(mid);
    state.venues[index].spread = Some(spread);
    state.venues[index].depth_near_mid = depth;
    state.venues[index].last_mid_update_ms = Some(update_time);
    state.venues[index].status = VenueStatus::Healthy;
}

/// Approximately equal for floating point comparison.
pub fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() < tolerance
}

/// Check if a value is finite (not NaN or infinite).
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

// ============================================================================
// NORMALIZED OUTPUT FOR GOLDEN COMPARISON
// ============================================================================

use serde::{Deserialize, Serialize};

/// Normalized representation of a hedge intent for golden comparison.
/// All fields are quantized to avoid floating-point micro-differences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedIntent {
    pub venue_index: usize,
    pub side: String,
    /// Size quantized to 9 decimal places (stored as integer nanoTAO).
    pub size_nano: i64,
    /// Price quantized to 9 decimal places.
    pub price_nano: i64,
}

impl NormalizedIntent {
    /// Create from raw intent values, quantizing floats.
    pub fn from_raw(venue_index: usize, side: &str, size: f64, price: f64) -> Self {
        NormalizedIntent {
            venue_index,
            side: side.to_string(),
            size_nano: (size * 1e9).round() as i64,
            price_nano: (price * 1e9).round() as i64,
        }
    }

    /// Convert back to approximate float values (for display).
    pub fn size_f64(&self) -> f64 {
        self.size_nano as f64 / 1e9
    }

    pub fn price_f64(&self) -> f64 {
        self.price_nano as f64 / 1e9
    }
}

/// Normalized representation of a hedge plan for golden comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedPlan {
    /// Desired delta quantized to 9 decimal places.
    pub desired_delta_nano: i64,
    /// Sorted intents for deterministic comparison.
    pub intents: Vec<NormalizedIntent>,
}

impl NormalizedPlan {
    /// Create an empty plan (no hedge action).
    pub fn empty() -> Self {
        NormalizedPlan {
            desired_delta_nano: 0,
            intents: Vec::new(),
        }
    }

    /// Create from a list of intents and desired delta.
    pub fn from_intents(desired_delta: f64, mut intents: Vec<NormalizedIntent>) -> Self {
        // Sort deterministically: by (venue_index, side, size_nano descending)
        intents.sort_by(|a, b| {
            let idx_cmp = a.venue_index.cmp(&b.venue_index);
            if idx_cmp != std::cmp::Ordering::Equal {
                return idx_cmp;
            }
            let side_cmp = a.side.cmp(&b.side);
            if side_cmp != std::cmp::Ordering::Equal {
                return side_cmp;
            }
            // Size descending (larger first)
            b.size_nano.cmp(&a.size_nano)
        });

        NormalizedPlan {
            desired_delta_nano: (desired_delta * 1e9).round() as i64,
            intents,
        }
    }
}

// ============================================================================
// GOLDEN VECTOR TYPES
// ============================================================================

/// A golden test vector with input configuration and expected output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenVector {
    /// Human-readable description of what this test case covers.
    pub description: String,
    /// Input configuration.
    pub input: GoldenInput,
    /// Expected normalized output.
    pub expected: NormalizedPlan,
}

/// Input configuration for a golden vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenInput {
    pub q_global_tao: f64,
    pub fair_value: f64,
    pub vol_ratio: f64,
    pub max_step_tao: f64,
    pub max_venue_tao_per_tick: f64,
    pub band_base_tao: f64,
    pub k_hedge: f64,
    pub margin_safety_buffer: f64,
    pub max_leverage: f64,
    pub chunk_size_tao: f64,
    pub chunk_convexity_cost_bps: f64,
    pub venues: Vec<GoldenVenueInput>,
}

/// Per-venue input for a golden vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenVenueInput {
    pub mid: f64,
    pub spread: f64,
    pub depth_usd: f64,
    pub position_tao: f64,
    pub margin_available_usd: f64,
    pub dist_liq_sigma: f64,
    pub funding_8h: f64,
    pub taker_fee_bps: f64,
}

impl GoldenInput {
    /// Build a Config from this golden input.
    pub fn build_config(&self) -> Config {
        let mut cfg = Config::default();

        cfg.hedge.max_step_tao = self.max_step_tao;
        cfg.hedge.max_venue_tao_per_tick = self.max_venue_tao_per_tick;
        cfg.hedge.band_base_tao = self.band_base_tao;
        cfg.hedge.band_vol_mult = 0.0; // Fixed for golden tests
        cfg.hedge.k_hedge = self.k_hedge;
        cfg.hedge.margin_safety_buffer = self.margin_safety_buffer;
        cfg.hedge.max_leverage = self.max_leverage;
        cfg.hedge.chunk_size_tao = self.chunk_size_tao;
        cfg.hedge.chunk_convexity_cost_bps = self.chunk_convexity_cost_bps;
        cfg.hedge.min_depth_usd = 100.0;
        cfg.hedge.depth_fraction = 0.5;

        // Zero out variable costs for predictability
        cfg.hedge.funding_weight = 0.0;
        cfg.hedge.basis_weight = 0.0;
        cfg.hedge.frag_penalty = 0.0;
        cfg.hedge.liq_penalty_scale = 0.0;
        cfg.hedge.slippage_buffer = 0.0;

        for (i, v) in self.venues.iter().enumerate() {
            if i < cfg.venues.len() {
                cfg.venues[i].taker_fee_bps = v.taker_fee_bps;
            }
        }

        cfg
    }

    /// Build a GlobalState from this golden input.
    ///
    /// Note: q_global_tao is derived from venue positions after recompute_after_fills.
    /// To get the desired q_global_tao, we put the position on venue 0 if venue
    /// positions don't already sum to the target.
    pub fn build_state(&self, cfg: &Config) -> GlobalState {
        let mut state = GlobalState::new(cfg);

        state.fair_value = Some(self.fair_value);
        state.fair_value_prev = self.fair_value;
        state.vol_ratio_clipped = self.vol_ratio;

        for (i, v) in self.venues.iter().enumerate() {
            if i >= state.venues.len() {
                break;
            }

            state.venues[i].mid = Some(v.mid);
            state.venues[i].spread = Some(v.spread);
            state.venues[i].depth_near_mid = v.depth_usd;
            state.venues[i].position_tao = v.position_tao;
            state.venues[i].margin_available_usd = v.margin_available_usd;
            state.venues[i].dist_liq_sigma = v.dist_liq_sigma;
            state.venues[i].funding_8h = v.funding_8h;
            state.venues[i].status = VenueStatus::Healthy;
            state.venues[i].last_mid_update_ms = Some(0);
        }

        // Calculate current sum of venue positions
        let venue_position_sum: f64 = self.venues.iter().map(|v| v.position_tao).sum();

        // If venue positions don't sum to q_global_tao, add the difference to venue 0
        // This ensures q_global_tao matches expectations after recompute_after_fills
        if (venue_position_sum - self.q_global_tao).abs() > 1e-9 && !self.venues.is_empty() {
            state.venues[0].position_tao = self.q_global_tao
                - self
                    .venues
                    .iter()
                    .skip(1)
                    .map(|v| v.position_tao)
                    .sum::<f64>();
        }

        state.recompute_after_fills(cfg);
        state
    }
}
