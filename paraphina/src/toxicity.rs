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

            // Compute instantaneous toxicity from markout:
            // - markout >= 0: favorable fill, tox_instant = 0
            // - markout < 0: adverse fill, tox_instant = clamp(-markout / scale, 0, 1)
            let tox_instant: f64 = if markout >= 0.0 {
                0.0
            } else {
                ((-markout) / markout_scale).clamp(0.0, 1.0)
            };

            // EWMA update: tox = (1 - alpha) * tox + alpha * tox_instant
            venue.toxicity = one_minus_alpha * venue.toxicity + alpha * tox_instant;

            // Also update markout EWMA for telemetry
            venue.markout_ewma_usd_per_tao =
                one_minus_alpha * venue.markout_ewma_usd_per_tao + alpha * markout;
        }

        // --- 2) Fallback: if no mid or no depth, apply legacy vol-based toxicity ---
        // This ensures venues with missing book data are still penalized.
        if venue.mid.is_none() || venue.depth_near_mid <= 0.0 {
            // No valid book data -> treat as highly toxic
            venue.toxicity = 1.0;
        } else if vol_tox_scale > 0.0 && sigma_eff > 0.0 {
            // Optionally blend in volatility-based feature for extra signal
            // (only if local vol is significantly elevated)
            let local: f64 = venue.local_vol_short.max(sigma_min);
            let ratio: f64 = local / sigma_eff;

            let raw: f64 = (ratio - 1.0) / vol_tox_scale;
            let f_vol: f64 = raw.clamp(0.0, 1.0);

            // Take max of current toxicity and vol-based feature
            // This provides a floor when markout data is sparse
            venue.toxicity = venue.toxicity.max(f_vol);
        }

        // --- 3) Clamp final toxicity to [0, 1] ---
        venue.toxicity = venue.toxicity.clamp(0.0, 1.0);

        // --- 4) Set venue health status based on toxicity thresholds ---
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
        });

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
        });

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
        });

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
        });

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // With alpha=0.5: tox = 0.5 * 0.7 + 0.5 * 1.0 = 0.85
        // Should be >= tox_high_threshold (0.8) => Disabled
        assert_eq!(state.venues[0].status, VenueStatus::Disabled);
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
        });

        update_toxicity_and_health(&mut state, &cfg, 1000);

        // Adverse markout for sell = price - mid_now = 100 - 102 = -2
        // tox_instant = clamp(2 / 1, 0, 1) = 1.0
        assert!(
            state.venues[0].toxicity >= 0.4,
            "Adverse sell markout should increase toxicity, got {}",
            state.venues[0].toxicity
        );
    }
}
