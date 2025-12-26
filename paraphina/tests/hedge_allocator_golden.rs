// tests/hedge_allocator_golden.rs
//
// Golden vector tests for the hedge allocator.
// These tests compare actual outputs against pre-computed expected results
// to protect against regressions in determinism, tie-breaking, and margin-cap behavior.
//
// Golden vectors are stored in tests/golden/hedge_allocator_vectors.json
//
// To update golden vectors intentionally:
// 1. Run tests with UPDATE_GOLDEN_VECTORS=1 to regenerate expected outputs
// 2. Review the diff carefully
// 3. Commit the updated vectors with a clear explanation

#[path = "hedge_testkit.rs"]
mod hedge_testkit;

use hedge_testkit::{
    GoldenInput, GoldenVector, GoldenVenueInput, NormalizedIntent, NormalizedPlan,
};
use paraphina::hedge::compute_hedge_plan;
use paraphina::types::Side;

// ============================================================================
// GOLDEN VECTORS (embedded from JSON)
// ============================================================================

const GOLDEN_VECTORS_JSON: &str = include_str!("golden/hedge_allocator_vectors.json");

/// Load golden vectors from the embedded JSON.
fn load_golden_vectors() -> Vec<GoldenVector> {
    let value: serde_json::Value =
        serde_json::from_str(GOLDEN_VECTORS_JSON).expect("Failed to parse golden vectors JSON");

    let vectors = value["vectors"]
        .as_array()
        .expect("Golden vectors should be an array");

    vectors
        .iter()
        .map(|v| {
            let description = v["description"].as_str().unwrap_or("unknown").to_string();

            let input = parse_golden_input(&v["input"]);
            let expected = parse_expected_plan(&v["expected"]);

            GoldenVector {
                description,
                input,
                expected,
            }
        })
        .collect()
}

fn parse_golden_input(v: &serde_json::Value) -> GoldenInput {
    let venues: Vec<GoldenVenueInput> = v["venues"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|vv| GoldenVenueInput {
            mid: vv["mid"].as_f64().unwrap_or(100.0),
            spread: vv["spread"].as_f64().unwrap_or(0.5),
            depth_usd: vv["depth_usd"].as_f64().unwrap_or(100000.0),
            position_tao: vv["position_tao"].as_f64().unwrap_or(0.0),
            margin_available_usd: vv["margin_available_usd"].as_f64().unwrap_or(100000.0),
            dist_liq_sigma: vv["dist_liq_sigma"].as_f64().unwrap_or(10.0),
            funding_8h: vv["funding_8h"].as_f64().unwrap_or(0.0),
            taker_fee_bps: vv["taker_fee_bps"].as_f64().unwrap_or(5.0),
        })
        .collect();

    GoldenInput {
        q_global_tao: v["q_global_tao"].as_f64().unwrap_or(0.0),
        fair_value: v["fair_value"].as_f64().unwrap_or(100.0),
        vol_ratio: v["vol_ratio"].as_f64().unwrap_or(1.0),
        max_step_tao: v["max_step_tao"].as_f64().unwrap_or(10.0),
        max_venue_tao_per_tick: v["max_venue_tao_per_tick"].as_f64().unwrap_or(10.0),
        band_base_tao: v["band_base_tao"].as_f64().unwrap_or(1.0),
        k_hedge: v["k_hedge"].as_f64().unwrap_or(0.5),
        margin_safety_buffer: v["margin_safety_buffer"].as_f64().unwrap_or(0.95),
        max_leverage: v["max_leverage"].as_f64().unwrap_or(10.0),
        chunk_size_tao: v["chunk_size_tao"].as_f64().unwrap_or(5.0),
        chunk_convexity_cost_bps: v["chunk_convexity_cost_bps"].as_f64().unwrap_or(0.0),
        venues,
    }
}

fn parse_expected_plan(v: &serde_json::Value) -> NormalizedPlan {
    let intents: Vec<NormalizedIntent> = v["intents"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|i| NormalizedIntent {
            venue_index: i["venue_index"].as_u64().unwrap_or(0) as usize,
            side: i["side"].as_str().unwrap_or("Sell").to_string(),
            size_nano: i["size_nano"].as_i64().unwrap_or(0),
            price_nano: i["price_nano"].as_i64().unwrap_or(0),
        })
        .collect();

    NormalizedPlan {
        desired_delta_nano: v["desired_delta_nano"].as_i64().unwrap_or(0),
        intents,
    }
}

// ============================================================================
// HELPER: Convert actual plan to normalized form
// ============================================================================

fn normalize_plan(plan: Option<paraphina::hedge::HedgePlan>) -> NormalizedPlan {
    match plan {
        None => NormalizedPlan::empty(),
        Some(p) => {
            let intents: Vec<NormalizedIntent> = p
                .allocations
                .iter()
                .map(|a| {
                    let side = match a.side {
                        Side::Buy => "Buy",
                        Side::Sell => "Sell",
                    };
                    NormalizedIntent::from_raw(a.venue_index, side, a.size, a.est_price)
                })
                .collect();

            NormalizedPlan::from_intents(p.desired_delta, intents)
        }
    }
}

// ============================================================================
// GOLDEN VECTOR TESTS
// ============================================================================

/// Main test that runs all golden vectors.
#[test]
fn golden_vector_tests() {
    let vectors = load_golden_vectors();

    assert!(
        !vectors.is_empty(),
        "No golden vectors loaded - check JSON file"
    );

    let mut failures = Vec::new();

    for (i, vector) in vectors.iter().enumerate() {
        let cfg = vector.input.build_config();
        let state = vector.input.build_state(&cfg);

        let actual_plan = compute_hedge_plan(&cfg, &state, 0);
        let actual_normalized = normalize_plan(actual_plan);

        // Compare normalized plans
        if actual_normalized != vector.expected {
            failures.push(format!(
                "\n=== Golden Vector {} FAILED ===\n\
                 Description: {}\n\
                 Expected: {:#?}\n\
                 Actual:   {:#?}\n\
                 Input: {:#?}",
                i, vector.description, vector.expected, actual_normalized, vector.input
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} golden vector test(s) failed:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

/// Test that each golden vector is deterministic (same result on multiple runs).
#[test]
fn golden_vectors_deterministic() {
    let vectors = load_golden_vectors();

    for (i, vector) in vectors.iter().enumerate() {
        let cfg = vector.input.build_config();
        let state = vector.input.build_state(&cfg);

        let plan1 = compute_hedge_plan(&cfg, &state, 0);
        let plan2 = compute_hedge_plan(&cfg, &state, 0);
        let plan3 = compute_hedge_plan(&cfg, &state, 0);

        let norm1 = normalize_plan(plan1);
        let norm2 = normalize_plan(plan2);
        let norm3 = normalize_plan(plan3);

        assert_eq!(
            norm1, norm2,
            "Golden vector {}: Non-deterministic result (run1 vs run2)\n\
             Description: {}",
            i, vector.description
        );
        assert_eq!(
            norm1, norm3,
            "Golden vector {}: Non-deterministic result (run1 vs run3)\n\
             Description: {}",
            i, vector.description
        );
    }
}

// ============================================================================
// INDIVIDUAL GOLDEN VECTOR TESTS (for specific coverage)
// ============================================================================

/// Test: Basic long position produces sell hedge.
#[test]
fn golden_basic_long_hedge() {
    let input = GoldenInput {
        q_global_tao: 10.0,
        fair_value: 100.0,
        vol_ratio: 1.0,
        max_step_tao: 10.0,
        max_venue_tao_per_tick: 10.0,
        band_base_tao: 1.0,
        k_hedge: 0.5,
        margin_safety_buffer: 0.95,
        max_leverage: 10.0,
        chunk_size_tao: 5.0,
        chunk_convexity_cost_bps: 0.0,
        venues: vec![GoldenVenueInput {
            mid: 100.0,
            spread: 0.5,
            depth_usd: 100000.0,
            position_tao: 0.0,
            margin_available_usd: 100000.0,
            dist_liq_sigma: 10.0,
            funding_8h: 0.0,
            taker_fee_bps: 5.0,
        }],
    };

    let cfg = input.build_config();
    let state = input.build_state(&cfg);

    let plan = compute_hedge_plan(&cfg, &state, 0);
    assert!(plan.is_some(), "Should produce a hedge plan");

    let plan = plan.unwrap();
    assert!(
        plan.desired_delta > 0.0,
        "Long position should have positive delta (sell)"
    );
    assert!(
        !plan.allocations.is_empty(),
        "Should have at least one allocation"
    );
    assert_eq!(
        plan.allocations[0].side,
        Side::Sell,
        "Should be selling to reduce long"
    );
}

/// Test: Basic short position produces buy hedge.
#[test]
fn golden_basic_short_hedge() {
    let input = GoldenInput {
        q_global_tao: -10.0,
        fair_value: 100.0,
        vol_ratio: 1.0,
        max_step_tao: 10.0,
        max_venue_tao_per_tick: 10.0,
        band_base_tao: 1.0,
        k_hedge: 0.5,
        margin_safety_buffer: 0.95,
        max_leverage: 10.0,
        chunk_size_tao: 5.0,
        chunk_convexity_cost_bps: 0.0,
        venues: vec![GoldenVenueInput {
            mid: 100.0,
            spread: 0.5,
            depth_usd: 100000.0,
            position_tao: 0.0,
            margin_available_usd: 100000.0,
            dist_liq_sigma: 10.0,
            funding_8h: 0.0,
            taker_fee_bps: 5.0,
        }],
    };

    let cfg = input.build_config();
    let state = input.build_state(&cfg);

    let plan = compute_hedge_plan(&cfg, &state, 0);
    assert!(plan.is_some(), "Should produce a hedge plan");

    let plan = plan.unwrap();
    assert!(
        plan.desired_delta < 0.0,
        "Short position should have negative delta (buy)"
    );
    assert!(
        !plan.allocations.is_empty(),
        "Should have at least one allocation"
    );
    assert_eq!(
        plan.allocations[0].side,
        Side::Buy,
        "Should be buying to cover short"
    );
}

/// Test: Position inside deadband produces no hedge.
#[test]
fn golden_deadband() {
    let input = GoldenInput {
        q_global_tao: 0.5, // Inside band_base_tao of 1.0
        fair_value: 100.0,
        vol_ratio: 1.0,
        max_step_tao: 10.0,
        max_venue_tao_per_tick: 10.0,
        band_base_tao: 1.0,
        k_hedge: 0.5,
        margin_safety_buffer: 0.95,
        max_leverage: 10.0,
        chunk_size_tao: 5.0,
        chunk_convexity_cost_bps: 0.0,
        venues: vec![GoldenVenueInput {
            mid: 100.0,
            spread: 0.5,
            depth_usd: 100000.0,
            position_tao: 0.0,
            margin_available_usd: 100000.0,
            dist_liq_sigma: 10.0,
            funding_8h: 0.0,
            taker_fee_bps: 5.0,
        }],
    };

    let cfg = input.build_config();
    let state = input.build_state(&cfg);

    let plan = compute_hedge_plan(&cfg, &state, 0);
    assert!(
        plan.is_none(),
        "Position in deadband should produce no hedge"
    );
}

/// Test: Margin constraint behavior when one venue has zero margin.
///
/// Note: In this test, venue 0 gets the position to make q_global_tao work out.
/// This means venue 0 has position=10 and is SELLING (reducing exposure), so
/// the zero margin doesn't block the trade. This correctly tests that margin
/// constraints only apply when INCREASING exposure, not when reducing it.
#[test]
fn golden_zero_margin_skipped() {
    let input = GoldenInput {
        q_global_tao: 10.0,
        fair_value: 100.0,
        vol_ratio: 1.0,
        max_step_tao: 10.0,
        max_venue_tao_per_tick: 10.0,
        band_base_tao: 1.0,
        k_hedge: 0.5,
        margin_safety_buffer: 1.0,
        max_leverage: 10.0,
        chunk_size_tao: 5.0,
        chunk_convexity_cost_bps: 0.0,
        venues: vec![
            GoldenVenueInput {
                mid: 100.0,
                spread: 0.5,
                depth_usd: 100000.0,
                position_tao: 0.0, // Will be set to 10.0 to make q_global_tao work
                margin_available_usd: 0.0, // Zero margin
                dist_liq_sigma: 10.0,
                funding_8h: 0.0,
                taker_fee_bps: 5.0,
            },
            GoldenVenueInput {
                mid: 100.0,
                spread: 0.5,
                depth_usd: 100000.0,
                position_tao: 0.0,
                margin_available_usd: 100000.0, // Ample margin
                dist_liq_sigma: 10.0,
                funding_8h: 0.0,
                taker_fee_bps: 5.0,
            },
        ],
    };

    let cfg = input.build_config();
    let state = input.build_state(&cfg);

    let plan = compute_hedge_plan(&cfg, &state, 0);
    assert!(plan.is_some(), "Should produce a hedge plan");

    let plan = plan.unwrap();

    // Venue 0 has the position (10 TAO) and is selling to reduce it.
    // Since selling reduces exposure (10 -> 5), zero margin doesn't block the trade.
    // This is correct behavior: margin constraints only apply when INCREASING exposure.
    let v0_alloc: f64 = plan
        .allocations
        .iter()
        .filter(|a| a.venue_index == 0)
        .map(|a| a.size)
        .sum();

    // Venue 0 should get allocation (reducing existing position is allowed)
    assert!(
        v0_alloc > 0.0,
        "Venue with existing position should be allowed to reduce exposure, got {v0_alloc}"
    );
}

/// Test: Max step cap is respected.
#[test]
fn golden_max_step_respected() {
    let input = GoldenInput {
        q_global_tao: 100.0, // Large position
        fair_value: 100.0,
        vol_ratio: 1.0,
        max_step_tao: 5.0, // But max step is only 5
        max_venue_tao_per_tick: 10.0,
        band_base_tao: 1.0,
        k_hedge: 1.0,
        margin_safety_buffer: 0.95,
        max_leverage: 10.0,
        chunk_size_tao: 5.0,
        chunk_convexity_cost_bps: 0.0,
        venues: vec![GoldenVenueInput {
            mid: 100.0,
            spread: 0.5,
            depth_usd: 100000.0,
            position_tao: 0.0,
            margin_available_usd: 100000.0,
            dist_liq_sigma: 10.0,
            funding_8h: 0.0,
            taker_fee_bps: 5.0,
        }],
    };

    let cfg = input.build_config();
    let state = input.build_state(&cfg);

    let plan = compute_hedge_plan(&cfg, &state, 0);
    assert!(plan.is_some(), "Should produce a hedge plan");

    let plan = plan.unwrap();
    let total: f64 = plan.allocations.iter().map(|a| a.size).sum();

    assert!(
        total <= 5.0 + 0.01,
        "Total allocation ({total}) should respect max_step_tao (5.0)"
    );
}

/// Test: Output is always sorted by venue_index.
#[test]
fn golden_output_sorted() {
    let input = GoldenInput {
        q_global_tao: 25.0,
        fair_value: 100.0,
        vol_ratio: 1.0,
        max_step_tao: 20.0,
        max_venue_tao_per_tick: 8.0,
        band_base_tao: 1.0,
        k_hedge: 1.0,
        margin_safety_buffer: 0.95,
        max_leverage: 10.0,
        chunk_size_tao: 4.0,
        chunk_convexity_cost_bps: 0.0,
        venues: vec![
            GoldenVenueInput {
                mid: 100.0,
                spread: 0.5,
                depth_usd: 100000.0,
                position_tao: 0.0,
                margin_available_usd: 100000.0,
                dist_liq_sigma: 10.0,
                funding_8h: 0.0,
                taker_fee_bps: 5.0,
            },
            GoldenVenueInput {
                mid: 100.0,
                spread: 0.5,
                depth_usd: 100000.0,
                position_tao: 0.0,
                margin_available_usd: 100000.0,
                dist_liq_sigma: 10.0,
                funding_8h: 0.0,
                taker_fee_bps: 5.0,
            },
            GoldenVenueInput {
                mid: 100.0,
                spread: 0.5,
                depth_usd: 100000.0,
                position_tao: 0.0,
                margin_available_usd: 100000.0,
                dist_liq_sigma: 10.0,
                funding_8h: 0.0,
                taker_fee_bps: 5.0,
            },
        ],
    };

    let cfg = input.build_config();
    let state = input.build_state(&cfg);

    let plan = compute_hedge_plan(&cfg, &state, 0);
    assert!(plan.is_some(), "Should produce a hedge plan");

    let plan = plan.unwrap();

    // Verify sorted by venue_index
    for i in 1..plan.allocations.len() {
        assert!(
            plan.allocations[i - 1].venue_index <= plan.allocations[i].venue_index,
            "Allocations should be sorted by venue_index"
        );
    }
}
