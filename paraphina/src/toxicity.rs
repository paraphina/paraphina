// src/toxicity.rs
//
// Per-venue markout-based toxicity scoring (v2 - whitepaper advancement).
//
// Toxicity is now driven by realized markouts from fills:
//
// 1. When a fill occurs, we record a pending markout evaluation scheduled
//    at t_fill + markout_horizon_ms.
//
// 2. At evaluation time, we compute:
//    - Buy:  markout = mid_now - fill_price
//    - Sell: markout = fill_price - mid_now
//
// 3. Convert markout to instantaneous toxicity:
//    - if markout >= 0 => tox_instant = 0 (favorable fill)
//    - else tox_instant = clamp((-markout) / markout_scale_usd_per_tao, 0, 1)
//
// 4. Update EWMA toxicity:
//    tox = (1 - alpha) * tox + alpha * tox_instant
//
// 5. Health gating based on toxicity thresholds:
//    - tox >= tox_high_threshold => Disabled
//    - tox >= tox_med_threshold  => Warning
//    - else                      => Healthy
//
// The legacy volatility-based toxicity feature is retained as a fallback
// when no mid/depth data is available.
//
// Ablation support:
// - disable_toxicity_gate: All venues remain Healthy regardless of toxicity score

use crate::config::Config;
use crate::sim_eval::AblationSet;
use crate::state::GlobalState;
use crate::types::{Side, TimestampMs, VenueStatus};

/// Update per-venue toxicity scores and health status.
///
/// This function:
/// 1. Processes pending markouts whose evaluation time has arrived.
/// 2. Updates toxicity EWMA based on markout results.
/// 3. Falls back to volatility-based toxicity when no book data exists.
/// 4. Sets venue health status based on toxicity thresholds.
pub fn update_toxicity_and_health(state: &mut GlobalState, cfg: &Config, now_ms: TimestampMs) {
    update_toxicity_and_health_impl::<false>(state, cfg, now_ms);
}

/// Update per-venue toxicity scores and health status with ablation support.
///
/// Same as update_toxicity_and_health, but with ablation support:
/// - disable_toxicity_gate: All venues remain Healthy regardless of toxicity score
///
/// # Optimization Notes
///
/// This function is optimized for the hot path:
///
/// - **Single-pass processing**: Pending markouts are drained and processed inline,
///   with no intermediate staging buffer. Each markout updates toxicity immediately
///   after being popped.
/// - **Hoisted config lookups**: Config values (alpha, scale, thresholds) are read once
///   before the venue loop, avoiding repeated struct access in the hot path.
/// - **Lazy mid resolution**: The venue.mid field is only checked/validated once per venue
///   per tick, and only after the first ready markout is popped. This avoids hot-path
///   overhead when no markouts are ready.
/// - **Markout tox_instant fastpath** (Opt19): Branchy logic avoids division/clamp when
///   the result is known: markout >= 0 yields 0, adverse >= scale yields 1, otherwise
///   a single division suffices (no clamp needed).
/// - **Vol fallback fastpath** (Opt19): When local_vol <= sigma_eff, f_vol is 0 and we
///   skip the ratio division entirely. When local_vol > sigma_eff, the raw value is
///   guaranteed non-negative so only the upper bound (min 1) is applied.
///
/// # Determinism Guarantees
///
/// Processing order is deterministic: venues are iterated in stable order, and for each
/// venue, markouts are processed in FIFO order from the front of pending_markouts.
/// Floating-point operations maintain identical evaluation order to preserve bit-exact results.
pub fn update_toxicity_and_health_with_ablations(
    state: &mut GlobalState,
    cfg: &Config,
    now_ms: TimestampMs,
    ablations: &AblationSet,
) {
    if ablations.disable_toxicity_gate() {
        update_toxicity_and_health_impl::<true>(state, cfg, now_ms);
    } else {
        update_toxicity_and_health_impl::<false>(state, cfg, now_ms);
    }
}

fn update_toxicity_and_health_impl<const DISABLE_TOX_GATE: bool>(
    state: &mut GlobalState,
    cfg: &Config,
    now_ms: TimestampMs,
) {
    let tox_cfg = &cfg.toxicity;
    let vol_cfg = &cfg.volatility;
    let shadow_mode = std::env::var("PARAPHINA_TRADE_MODE")
        .ok()
        .map(|v| {
            matches!(
                v.to_ascii_lowercase().as_str(),
                "shadow" | "safe" | "s" | "paper" | "p" | "testnet" | "tn" | "t"
            )
        })
        .unwrap_or(false);

    // Hoist config lookups outside the venue loop to avoid repeated struct access.
    let markout_scale = tox_cfg.markout_scale_usd_per_tao.max(1e-9);
    let alpha = tox_cfg.markout_alpha.clamp(0.0, 1.0);
    let one_minus_alpha = 1.0 - alpha;
    let vol_tox_scale = tox_cfg.vol_tox_scale;
    let tox_high_threshold = tox_cfg.tox_high_threshold;
    let tox_med_threshold = tox_cfg.tox_med_threshold;
    let sigma_min = vol_cfg.sigma_min;

    // Keep sigma_eff away from zero so ratios are well-defined.
    let sigma_eff: f64 = state.sigma_eff.max(sigma_min);

    for venue in state.venues.iter_mut() {
        // --- 1) Process pending markouts in a single pass ---
        // Opt17: Use cached next_eval_ms to skip deque access when no markouts are ready.
        if venue.pending_markouts_next_eval_ms <= now_ms {
            // Lazy mid resolution: only check venue.mid once we pop the first ready markout.
            let mut mid_checked = false;
            let mut have_mid = false;
            let mut mid: f64 = 0.0;

            // Drain and process all markouts where t_eval_ms <= now_ms in FIFO order.
            // Each markout is processed inline - no intermediate staging buffer.
            while let Some(front) = venue.pending_markouts.front() {
                if front.t_eval_ms > now_ms {
                    break; // Not yet ready
                }

                let pm = venue.pending_markouts.pop_front().unwrap();

                // Lazy mid resolution: only check venue.mid once per venue per tick
                if !mid_checked {
                    mid_checked = true;
                    match venue.mid {
                        Some(m) if m.is_finite() && m > 0.0 => {
                            mid = m;
                            have_mid = true;
                        }
                        _ => {}
                    }
                }

                // Skip EWMA update if no valid mid, but still drain the markout
                if !have_mid {
                    continue;
                }

                // Compute markout in USD/TAO
                // Buy: we want price to go UP after buying, so markout = mid_now - fill_price
                // Sell: we want price to go DOWN after selling, so markout = fill_price - mid_now
                let markout = match pm.side {
                    Side::Buy => mid - pm.price,
                    Side::Sell => pm.price - mid,
                };

                // Attach markout to the corresponding recent fill (if present).
                venue.apply_markout_to_fill(pm.fill_seq, markout);

                // Compute instantaneous toxicity from markout:
                // - markout >= 0: favorable fill, tox_instant = 0
                // - markout < 0: adverse fill, tox_instant in (0, 1]
                // Opt19: Branchy fastpath avoids division/clamp when result is known.
                let tox_instant: f64 = if markout >= 0.0 {
                    0.0
                } else {
                    let adverse = -markout;
                    if adverse >= markout_scale {
                        1.0
                    } else {
                        // 0 < adverse < markout_scale, so 0 < result < 1; no clamp needed.
                        adverse / markout_scale
                    }
                };

                // EWMA update: tox = (1 - alpha) * tox + alpha * tox_instant
                venue.toxicity = one_minus_alpha * venue.toxicity + alpha * tox_instant;

                // Also update markout EWMA for telemetry
                venue.markout_ewma_usd_per_tao =
                    one_minus_alpha * venue.markout_ewma_usd_per_tao + alpha * markout;
            }

            // Opt17: Refresh cached next eval time after draining matured markouts.
            venue.pending_markouts_next_eval_ms = venue
                .pending_markouts
                .front()
                .map_or(i64::MAX, |pm| pm.t_eval_ms);
        }

        // --- 2) Shadow-mode warmup override ---
        if shadow_mode
            && venue.last_mid_update_ms.is_some()
            && venue.mid.unwrap_or(0.0) > 0.0
            && venue.depth_near_mid > 0.0
        {
            venue.toxicity = 0.0;
        }

        // --- 3) Fallback: if no mid or prolonged no depth, apply legacy toxicity ---
        // This ensures venues with missing book data are still penalized,
        // while avoiding false disables from brief empty-side snapshots.
        let depth_zero_prolonged = venue.depth_near_mid <= 0.0
            && venue
                .last_mid_update_ms
                .map(|last_ms| {
                    now_ms.saturating_sub(last_ms) > cfg.toxicity.depth_fallback_grace_ms
                })
                .unwrap_or(true);
        if venue.mid.is_none() || depth_zero_prolonged {
            // No valid book data -> treat as highly toxic
            venue.toxicity = 1.0;
        } else if !shadow_mode && vol_tox_scale > 0.0 && sigma_eff > 0.0 {
            // Optionally blend in volatility-based feature for extra signal
            // (only if local vol is significantly elevated)
            let local: f64 = venue.local_vol_short.max(sigma_min);

            // Opt19: Fastpath - if local <= sigma_eff, ratio <= 1, so raw <= 0, so f_vol = 0.
            // Skip division entirely in this case.
            let f_vol: f64 = if local <= sigma_eff {
                0.0
            } else {
                let ratio: f64 = local / sigma_eff;
                let raw: f64 = (ratio - 1.0) / vol_tox_scale;
                // ratio > 1 implies raw > 0, so only upper bound needed.
                raw.min(1.0)
            };

            // Take max of current toxicity and vol-based feature
            // This provides a floor when markout data is sparse
            venue.toxicity = venue.toxicity.max(f_vol);
        }

        // --- 4) Clamp final toxicity to [0, 1] ---
        venue.toxicity = venue.toxicity.clamp(0.0, 1.0);

        // --- 5) Set venue health status based on toxicity thresholds ---
        // If disable_toxicity_gate ablation is active, all venues remain Healthy
        if DISABLE_TOX_GATE {
            venue.status = VenueStatus::Healthy;
        } else {
            venue.status = if venue.toxicity >= tox_high_threshold {
                VenueStatus::Disabled
            } else if venue.toxicity >= tox_med_threshold {
                VenueStatus::Warning
            } else {
                VenueStatus::Healthy
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::state::PendingMarkout;
    use crate::types::Side;
    use std::sync::Mutex;

    fn make_test_config() -> Config {
        let mut cfg = Config::default();
        // Set predictable toxicity thresholds for testing
        cfg.toxicity.tox_med_threshold = 0.5;
        cfg.toxicity.tox_high_threshold = 0.8;
        cfg.toxicity.markout_alpha = 0.5; // 50% blend for easier testing
        cfg.toxicity.markout_scale_usd_per_tao = 1.0; // $1 adverse = tox=1
        cfg.toxicity.markout_horizon_ms = 1000; // 1 second
        cfg
    }

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_favorable_markout_does_not_increase_toxicity() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        // Setup venue with valid mid
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.2; // Start with some toxicity

        // Add a pending markout for a BUY at 99, now mid is 100 (favorable!)
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 99.0, // Bought at 99
            fair_at_fill: 99.0,
            mid_at_fill: 99.0,
            fill_seq: 0,
        });
        venue.pending_markouts_next_eval_ms = 1000;

        // Run toxicity update at t=1000ms (markout should evaluate)
        update_toxicity_and_health(&mut state, &cfg, 1000);

        // Favorable markout = mid_now - price = 100 - 99 = +1
        // tox_instant = 0 (favorable)
        // tox = 0.5 * 0.2 + 0.5 * 0 = 0.1
        assert!(
            state.venues[0].toxicity < 0.2,
            "Favorable markout should decrease toxicity, got {}",
            state.venues[0].toxicity
        );
    }

    #[test]
    fn test_adverse_markout_increases_toxicity() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        // Setup venue with valid mid
        let venue = &mut state.venues[0];
        venue.mid = Some(98.0); // Price dropped!
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.0; // Start clean

        // Add a pending markout for a BUY at 100, now mid is 98 (adverse!)
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 100.0, // Bought at 100
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
            fill_seq: 0,
        });
        venue.pending_markouts_next_eval_ms = 1000;

        // Run toxicity update at t=1000ms
        update_toxicity_and_health(&mut state, &cfg, 1000);

        // Adverse markout = mid_now - price = 98 - 100 = -2
        // tox_instant = clamp(2 / 1, 0, 1) = 1.0
        // tox = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert!(
            state.venues[0].toxicity >= 0.4,
            "Adverse markout should increase toxicity, got {}",
            state.venues[0].toxicity
        );
    }

    #[test]
    fn test_markout_only_applies_after_horizon() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        // Setup venue
        let venue = &mut state.venues[0];
        venue.mid = Some(98.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.0;

        // Add pending markout that evaluates at t=1000
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 100.0,
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
            fill_seq: 0,
        });
        venue.pending_markouts_next_eval_ms = 1000;

        // Run at t=500 (before horizon)
        update_toxicity_and_health(&mut state, &cfg, 500);

        // Toxicity should remain low (no markout processed yet)
        assert!(
            state.venues[0].toxicity < 0.1,
            "Markout should not apply before horizon, got {}",
            state.venues[0].toxicity
        );

        // Markout should still be pending
        assert_eq!(state.venues[0].pending_markouts.len(), 1);

        // Now run at t=1000 (at horizon)
        update_toxicity_and_health(&mut state, &cfg, 1000);

        // Now toxicity should have increased
        assert!(
            state.venues[0].toxicity >= 0.4,
            "Markout should apply at horizon, got {}",
            state.venues[0].toxicity
        );

        // Markout should be consumed
        assert_eq!(state.venues[0].pending_markouts.len(), 0);
    }

    #[test]
    fn test_high_toxicity_disables_venue() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        // Setup venue
        let venue = &mut state.venues[0];
        venue.mid = Some(90.0); // Price crashed!
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.7; // Already elevated

        // Add very adverse markout
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 100.0,
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
            fill_seq: 0,
        });
        venue.pending_markouts_next_eval_ms = 1000;

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // With alpha=0.5: tox = 0.5 * 0.7 + 0.5 * 1.0 = 0.85
        // Should be >= tox_high_threshold (0.8) => Disabled
        assert_eq!(state.venues[0].status, VenueStatus::Disabled);
    }

    #[test]
    fn test_depth_zero_within_grace_does_not_force_toxicity() {
        let mut cfg = make_test_config();
        cfg.toxicity.depth_fallback_grace_ms = 500;
        let mut state = GlobalState::new(&cfg);

        let now_ms = 2_000;
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 0.0;
        venue.last_mid_update_ms = Some(1_800); // 200ms ago, within grace
        venue.toxicity = 0.2;

        update_toxicity_and_health(&mut state, &cfg, now_ms);

        assert!(
            state.venues[0].toxicity < 1.0,
            "Depth=0 within grace should not force toxicity to 1.0"
        );
        assert_ne!(
            state.venues[0].status,
            VenueStatus::Disabled,
            "Depth=0 within grace should not disable venue"
        );
    }

    #[test]
    fn test_depth_zero_beyond_grace_forces_toxicity() {
        let mut cfg = make_test_config();
        cfg.toxicity.depth_fallback_grace_ms = 500;
        let mut state = GlobalState::new(&cfg);

        let now_ms = 2_000;
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 0.0;
        venue.last_mid_update_ms = Some(1_000); // 1000ms ago, beyond grace
        venue.toxicity = 0.2;

        update_toxicity_and_health(&mut state, &cfg, now_ms);

        assert_eq!(
            state.venues[0].toxicity, 1.0,
            "Depth=0 beyond grace should force toxicity to 1.0"
        );
        assert_eq!(
            state.venues[0].status,
            VenueStatus::Disabled,
            "Depth=0 beyond grace should disable venue"
        );
    }

    #[test]
    fn test_sell_adverse_markout() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        // Setup venue
        let venue = &mut state.venues[0];
        venue.mid = Some(102.0); // Price went UP after we sold (adverse)
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.0;

        // Add pending markout for a SELL at 100
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Sell,
            size_tao: 1.0,
            price: 100.0, // Sold at 100
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
            fill_seq: 0,
        });
        venue.pending_markouts_next_eval_ms = 1000;

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // Adverse markout for sell = price - mid_now = 100 - 102 = -2
        // tox_instant = clamp(2 / 1, 0, 1) = 1.0
        assert!(
            state.venues[0].toxicity >= 0.4,
            "Adverse sell markout should increase toxicity, got {}",
            state.venues[0].toxicity
        );
    }

    /// Test FIFO processing: markouts with increasing maturity times are popped
    /// in order, and only those whose t_eval_ms <= now_ms are processed.
    #[test]
    fn test_fifo_pops_exactly_matured_subset() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        // Setup venue with valid mid
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.0;

        // Push 5 markouts with strictly increasing maturity times.
        // These represent fills at t=0,100,200,300,400 with horizon=1000.
        for i in 0..5 {
            venue.pending_markouts.push_back(PendingMarkout {
                t_fill_ms: i * 100,
                t_eval_ms: (i * 100) + 1000, // t_eval = 1000, 1100, 1200, 1300, 1400
                side: Side::Buy,
                size_tao: 1.0,
                price: 100.0,
                fair_at_fill: 100.0,
                mid_at_fill: 100.0,
                fill_seq: i as u64,
            });
        }
        venue.pending_markouts_next_eval_ms = 1000; // First entry's t_eval_ms

        assert_eq!(venue.pending_markouts.len(), 5);

        // At t=1150, markouts with t_eval_ms <= 1150 should be processed.
        // That's t_eval_ms = 1000 and 1100 (indices 0 and 1).
        update_toxicity_and_health(&mut state, &cfg, 1150);

        // Should have processed 2 markouts, leaving 3 pending.
        assert_eq!(
            state.venues[0].pending_markouts.len(),
            3,
            "Expected 3 pending markouts after processing at t=1150"
        );

        // Verify the remaining markouts are the correct ones (1200, 1300, 1400).
        let remaining: Vec<i64> = state.venues[0]
            .pending_markouts
            .iter()
            .map(|pm| pm.t_eval_ms)
            .collect();
        assert_eq!(remaining, vec![1200, 1300, 1400]);

        // Process one more at t=1200 (exactly at deadline).
        update_toxicity_and_health(&mut state, &cfg, 1200);
        assert_eq!(
            state.venues[0].pending_markouts.len(),
            2,
            "Expected 2 pending markouts after processing at t=1200"
        );

        // Process all remaining at t=2000.
        update_toxicity_and_health(&mut state, &cfg, 2000);
        assert_eq!(
            state.venues[0].pending_markouts.len(),
            0,
            "Expected 0 pending markouts after processing at t=2000"
        );
    }

    /// Test that multiple markouts with the same t_eval_ms are all processed
    /// when that time arrives.
    #[test]
    fn test_fifo_processes_same_time_markouts() {
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);

        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.0;

        // Push 3 markouts all with the same t_eval_ms (can happen if fills
        // occur at the same timestamp).
        for _ in 0..3 {
            venue.pending_markouts.push_back(PendingMarkout {
                t_fill_ms: 0,
                t_eval_ms: 1000,
                side: Side::Buy,
                size_tao: 1.0,
                price: 100.0,
                fair_at_fill: 100.0,
                mid_at_fill: 100.0,
                fill_seq: 0,
            });
        }

        // Add one more with a later time.
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 100,
            t_eval_ms: 1100,
            side: Side::Buy,
            size_tao: 1.0,
            price: 100.0,
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
            fill_seq: 3,
        });
        venue.pending_markouts_next_eval_ms = 1000; // First entry's t_eval_ms

        assert_eq!(venue.pending_markouts.len(), 4);

        // Process at t=1000: all 3 with t_eval_ms=1000 should be processed.
        update_toxicity_and_health(&mut state, &cfg, 1000);

        assert_eq!(
            state.venues[0].pending_markouts.len(),
            1,
            "Expected 1 pending markout (t_eval=1100) after processing at t=1000"
        );
        assert_eq!(
            state.venues[0].pending_markouts.front().unwrap().t_eval_ms,
            1100
        );
    }

    /// Test that FIFO order is preserved: earlier-added markouts are processed
    /// before later-added ones, maintaining deterministic behavior.
    #[test]
    fn test_fifo_preserves_insertion_order() {
        let mut cfg = make_test_config();
        // Use alpha=1.0 so toxicity fully reflects the most recent markout.
        cfg.toxicity.markout_alpha = 1.0;

        let mut state = GlobalState::new(&cfg);

        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.0;

        // Push two markouts at same t_eval_ms but different prices.
        // First: Buy at 100 (neutral, markout=0).
        // Second: Buy at 101 (adverse, markout=-1).
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 100.0, // markout = 100 - 100 = 0 → tox=0
            fair_at_fill: 100.0,
            mid_at_fill: 100.0,
            fill_seq: 0,
        });
        venue.pending_markouts.push_back(PendingMarkout {
            t_fill_ms: 0,
            t_eval_ms: 1000,
            side: Side::Buy,
            size_tao: 1.0,
            price: 101.0, // markout = 100 - 101 = -1 → tox=1
            fair_at_fill: 101.0,
            mid_at_fill: 101.0,
            fill_seq: 1,
        });
        venue.pending_markouts_next_eval_ms = 1000;

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // With alpha=1.0, toxicity should reflect the LAST processed markout.
        // FIFO order means first (price=100, tox=0) is processed, then
        // second (price=101, tox=1). Final toxicity should be 1.0.
        assert!(
            (state.venues[0].toxicity - 1.0).abs() < 1e-9,
            "Expected toxicity=1.0 (last markout), got {}",
            state.venues[0].toxicity
        );
    }

    /// Test Opt19 vol fallback fastpath: when local_vol <= sigma_eff, f_vol = 0
    /// and toxicity is not increased by the volatility feature.
    #[test]
    fn vol_fallback_local_le_sigma_eff_is_zero() {
        let mut cfg = make_test_config();
        // Ensure vol_tox_scale > 0 so the vol fallback branch is active.
        cfg.toxicity.vol_tox_scale = 0.5;
        cfg.volatility.sigma_min = 0.001;

        let mut state = GlobalState::new(&cfg);

        // Setup venue with valid mid and depth so it doesn't hit the missing-book fallback.
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.3; // Start with some toxicity

        // Set sigma_eff and local_vol_short such that local <= sigma_eff.
        state.sigma_eff = 0.2;
        state.venues[0].local_vol_short = 0.1; // 0.1 <= 0.2

        // No pending markouts, so only the vol fallback path runs.
        // pending_markouts_next_eval_ms defaults to i64::MAX, so the markout loop is skipped.

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // With local <= sigma_eff, f_vol = 0 (Opt19 fastpath).
        // toxicity = max(0.3, 0.0) = 0.3 (unchanged).
        assert!(
            (state.venues[0].toxicity - 0.3).abs() < 1e-9,
            "Expected toxicity to remain 0.3 when local_vol <= sigma_eff, got {}",
            state.venues[0].toxicity
        );
    }

    /// Test Opt19 vol fallback: when local_vol > sigma_eff, f_vol > 0
    /// and toxicity is raised to at least f_vol.
    #[test]
    fn vol_fallback_local_gt_sigma_eff_increases_toxicity_floor() {
        let mut cfg = make_test_config();
        // Ensure vol_tox_scale > 0 so the vol fallback branch is active.
        cfg.toxicity.vol_tox_scale = 0.5;
        cfg.volatility.sigma_min = 0.001;

        let mut state = GlobalState::new(&cfg);

        // Setup venue with valid mid and depth so it doesn't hit the missing-book fallback.
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(0.1);
        venue.depth_near_mid = 10000.0;
        venue.toxicity = 0.1; // Start with low toxicity

        // Set sigma_eff and local_vol_short such that local > sigma_eff.
        state.sigma_eff = 0.2;
        state.venues[0].local_vol_short = 0.25; // 0.25 > 0.2

        // No pending markouts, so only the vol fallback path runs.

        // Expected f_vol calculation:
        // local = max(0.25, 0.001) = 0.25
        // ratio = 0.25 / 0.2 = 1.25
        // raw = (1.25 - 1.0) / 0.5 = 0.5
        // f_vol = min(0.5, 1.0) = 0.5
        let expected_f_vol = 0.5;

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // toxicity = max(0.1, 0.5) = 0.5
        assert!(
            (state.venues[0].toxicity - expected_f_vol).abs() < 1e-9,
            "Expected toxicity to be raised to {} when local_vol > sigma_eff, got {}",
            expected_f_vol,
            state.venues[0].toxicity
        );
    }

    #[test]
    fn shadow_mode_warmup_clears_toxicity_with_book() {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("PARAPHINA_TRADE_MODE", "shadow");
        let cfg = make_test_config();
        let mut state = GlobalState::new(&cfg);
        let venue = &mut state.venues[0];
        venue.mid = Some(100.0);
        venue.spread = Some(1.0);
        venue.depth_near_mid = 500.0;
        venue.last_mid_update_ms = Some(1_000);
        venue.toxicity = 1.0;

        update_toxicity_and_health(&mut state, &cfg, 1_000);

        assert_eq!(state.venues[0].toxicity, 0.0);
        assert!(matches!(state.venues[0].status, VenueStatus::Healthy));
        std::env::remove_var("PARAPHINA_TRADE_MODE");
    }
}
