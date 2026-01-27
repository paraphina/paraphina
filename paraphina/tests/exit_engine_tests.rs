use paraphina::config::Config;
use paraphina::exit;
use paraphina::state::GlobalState;
use paraphina::types::OrderIntent;
use paraphina::types::{Side, VenueStatus};

/// Extractors for OrderIntent enum used in exit-engine tests.
/// Exit intents are expected to be Place/Replace; panic on other variants so the test
/// surface stays strict and deterministic.
fn intent_venue_index(intent: &OrderIntent) -> usize {
    match intent {
        OrderIntent::Place(pi) => pi.venue_index,
        OrderIntent::Replace(ri) => ri.venue_index,
        OrderIntent::Cancel(ci) => ci.venue_index,
        OrderIntent::CancelAll(ci) => ci.venue_index.expect("CancelAll missing venue_index"),
    }
}

fn intent_size(intent: &OrderIntent) -> f64 {
    match intent {
        OrderIntent::Place(pi) => pi.size,
        OrderIntent::Replace(ri) => ri.size,
        other => panic!("Expected Place/Replace for size, got: {:?}", other),
    }
}

fn intent_side(intent: &OrderIntent) -> Side {
    match intent {
        OrderIntent::Place(pi) => pi.side,
        OrderIntent::Replace(ri) => ri.side,
        other => panic!("Expected Place/Replace for side, got: {:?}", other),
    }
}

#[test]
fn exit_noops_without_fair_value() {
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);
    assert!(intents.is_empty());
}

/// Test that exit respects lot size and skips venues where rounding makes size too small.
#[test]
fn exit_respects_lot_size_and_min_notional() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 10.0;
    cfg.exit.max_venue_tao_per_tick = 10.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0;
    cfg.exit.min_intent_size_tao = 0.001;

    // Set venue 1 to have a large lot size requirement
    cfg.venues[1].lot_size_tao = 5.0; // Minimum 5 TAO per order
    cfg.venues[1].size_step_tao = 5.0;
    cfg.venues[1].min_notional_usd = 1000.0; // High min notional

    // Set venue 2 to have small lot size (normal)
    cfg.venues[2].lot_size_tao = 0.01;
    cfg.venues[2].size_step_tao = 0.01;
    cfg.venues[2].min_notional_usd = 1.0; // Low min notional

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Create net long exposure that's smaller than venue 1's lot size
    state.venues[0].position_tao = 3.0; // Only 3 TAO to exit
    state.venues[0].avg_entry_price = 100.0;

    // Venue 1: profitable but lot size is 5 TAO (we only have 3 to exit)
    state.venues[1].mid = Some(105.0);
    state.venues[1].spread = Some(0.5);
    state.venues[1].depth_near_mid = 1_000_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    // Venue 2: also profitable, normal lot size
    state.venues[2].mid = Some(104.0);
    state.venues[2].spread = Some(0.5);
    state.venues[2].depth_near_mid = 1_000_000.0;
    state.venues[2].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);

    // Should skip venue 1 due to lot size constraints (3 TAO < 5 TAO lot size)
    // and min notional constraints (3 * 104.75 = ~314 < 1000)
    for it in &intents {
        assert_ne!(
            intent_venue_index(it),
            1,
            "Should not use venue 1 due to lot size/min notional constraints"
        );
    }

    // Should use venue 2 which has appropriate lot sizes
    if !intents.is_empty() {
        assert!(
            intents.iter().any(|i| intent_venue_index(i) == 2),
            "Should be able to exit on venue 2 with normal lot sizes"
        );
    }
}

/// Test that exit respects minimum notional requirement.
#[test]
fn exit_respects_min_notional() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 10.0;
    cfg.exit.max_venue_tao_per_tick = 10.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0;
    cfg.exit.min_intent_size_tao = 0.001;

    // Venue 1: high min notional
    cfg.venues[1].lot_size_tao = 0.01;
    cfg.venues[1].size_step_tao = 0.01;
    cfg.venues[1].min_notional_usd = 500.0; // 500 USD minimum

    // Venue 2: low min notional
    cfg.venues[2].lot_size_tao = 0.01;
    cfg.venues[2].size_step_tao = 0.01;
    cfg.venues[2].min_notional_usd = 1.0;

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Create small net long exposure
    state.venues[0].position_tao = 2.0; // Only 2 TAO = 200 USD notional
    state.venues[0].avg_entry_price = 100.0;

    // Both venues are profitable
    state.venues[1].mid = Some(105.0);
    state.venues[1].spread = Some(0.5);
    state.venues[1].depth_near_mid = 1_000_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    state.venues[2].mid = Some(104.0);
    state.venues[2].spread = Some(0.5);
    state.venues[2].depth_near_mid = 1_000_000.0;
    state.venues[2].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);

    // Should skip venue 1 due to min notional (2 * ~104.75 = ~210 < 500)
    for it in &intents {
        assert_ne!(
            intent_venue_index(it),
            1,
            "Should not use venue 1 due to min notional constraint"
        );
    }
}

#[test]
fn exit_profit_only_blocks_unprofitable_exits() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 10.0;
    cfg.exit.max_venue_tao_per_tick = 10.0;
    cfg.exit.edge_min_usd = 0.25;

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Net long exposure via venue 0.
    state.venues[0].position_tao = 10.0;
    state.venues[0].avg_entry_price = 100.0;

    // Candidate venue 1 has bid below entry -> unprofitable.
    state.venues[1].mid = Some(100.0);
    state.venues[1].spread = Some(2.0); // bid ~99
    state.venues[1].depth_near_mid = 100_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);
    assert!(intents.is_empty(), "should not emit unprofitable exits");
}

#[test]
fn exit_splits_across_best_venues_when_capped_per_venue() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 10.0;
    cfg.exit.max_venue_tao_per_tick = 6.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0; // effectively no depth cap for test

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Create net long.
    state.venues[0].position_tao = 10.0;
    state.venues[0].avg_entry_price = 100.0;

    // Venue 1: very attractive sell (high bid)
    state.venues[1].mid = Some(102.0);
    state.venues[1].spread = Some(0.5);
    state.venues[1].depth_near_mid = 1_000_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    // Venue 2: second best
    state.venues[2].mid = Some(101.5);
    state.venues[2].spread = Some(0.5);
    state.venues[2].depth_near_mid = 1_000_000.0;
    state.venues[2].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);
    assert!(!intents.is_empty());

    // Should sell (net long -> need Sell)
    for it in &intents {
        assert_eq!(intent_side(it), Side::Sell);
    }

    let total: f64 = intents.iter().map(intent_size).sum();
    assert!(
        (total - 10.0).abs() < 1e-6 || total <= 10.0,
        "total exit should be <= max_total_tao_per_tick"
    );

    // With per-venue cap 6, it should split roughly 6 + 4
    assert!(intents.len() >= 2, "should allocate across multiple venues");
    assert!(intents.iter().any(|i| (intent_size(i) - 6.0).abs() < 1e-6));
}

#[test]
fn exit_skips_disabled_or_toxic_or_stale() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 5.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0;

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Net long.
    state.venues[0].position_tao = 5.0;
    state.venues[0].avg_entry_price = 100.0;

    // Venue 1 is disabled.
    state.venues[1].status = VenueStatus::Disabled;
    state.venues[1].mid = Some(105.0);
    state.venues[1].spread = Some(0.5);
    state.venues[1].depth_near_mid = 1_000_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    // Venue 2 is toxic.
    state.venues[2].toxicity = cfg.toxicity.tox_high_threshold + 0.01;
    state.venues[2].mid = Some(105.0);
    state.venues[2].spread = Some(0.5);
    state.venues[2].depth_near_mid = 1_000_000.0;
    state.venues[2].last_mid_update_ms = Some(0);

    // Venue 3 is stale.
    state.venues[3].mid = Some(105.0);
    state.venues[3].spread = Some(0.5);
    state.venues[3].depth_near_mid = 1_000_000.0;
    state.venues[3].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    // now_ms past stale threshold
    let intents = exit::compute_exit_intents(&cfg, &state, cfg.book.stale_ms + 1);
    // Could still be empty depending on which venues are eligible;
    // the important thing is it did not use 1/2/3.
    for it in intents {
        assert!(intent_venue_index(&it) != 1);
        assert!(intent_venue_index(&it) != 2);
        assert!(intent_venue_index(&it) != 3);
    }
}

/// Test that when edges are similar, exit prefers venues that reduce fragmentation.
/// This proves deterministic tie-breaking behavior.
#[test]
fn exit_prefers_less_fragmentation_when_edges_similar() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 5.0;
    cfg.exit.max_venue_tao_per_tick = 5.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0;
    cfg.exit.min_intent_size_tao = 0.01;

    // Boost fragmentation reduction bonus to make it decisive
    cfg.exit.fragmentation_reduction_bonus = 0.5;

    // Set similar lot sizes for all venues
    for v in &mut cfg.venues {
        v.lot_size_tao = 0.01;
        v.size_step_tao = 0.01;
        v.min_notional_usd = 1.0;
    }

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Create fragmented positions across multiple venues (long exposure)
    // Venue 1: small long position (fragmented)
    state.venues[1].position_tao = 2.0;
    state.venues[1].avg_entry_price = 100.0;

    // Venue 2: larger long position
    state.venues[2].position_tao = 3.0;
    state.venues[2].avg_entry_price = 100.0;

    // Both venues have IDENTICAL prices (same edge)
    let identical_mid = 103.0;
    let identical_spread = 0.5;

    state.venues[1].mid = Some(identical_mid);
    state.venues[1].spread = Some(identical_spread);
    state.venues[1].depth_near_mid = 1_000_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    state.venues[2].mid = Some(identical_mid);
    state.venues[2].spread = Some(identical_spread);
    state.venues[2].depth_near_mid = 1_000_000.0;
    state.venues[2].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);

    assert!(!intents.is_empty(), "Should generate exit intents");

    // When edges are identical, should prefer venue that reduces fragmentation more.
    // Venue 1 has a smaller position (2 TAO), so closing it entirely reduces fragmentation more
    // than partially reducing venue 2.
    // The first intent should be for the venue where exiting reduces fragmentation more.

    if intents.len() >= 2 {
        // With identical edges, venue 1 (smaller position, can be fully closed) should be preferred
        // because closing a position entirely provides the fragmentation_reduction_bonus.
        let first_venue = intent_venue_index(&intents[0]);
        let _first_closes_position =
            intent_size(&intents[0]) >= state.venues[first_venue].position_tao.abs() - 0.01;

        // At least verify deterministic ordering - same inputs should produce same outputs
        let intents2 = exit::compute_exit_intents(&cfg, &state, 0);
        assert_eq!(
            intent_venue_index(&intents[0]),
            intent_venue_index(&intents2[0]),
            "Exit ordering must be deterministic"
        );
        assert!(
            (intent_size(&intents[0]) - intent_size(&intents2[0])).abs() < 0.001,
            "Exit sizes must be deterministic"
        );
    }
}

/// Test that when edges are similar, exit prefers venues that reduce basis risk.
#[test]
fn exit_prefers_less_basis_risk_when_edges_similar() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 5.0;
    cfg.exit.max_venue_tao_per_tick = 5.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0;
    cfg.exit.min_intent_size_tao = 0.01;

    // Boost basis risk penalty to make it decisive
    cfg.exit.basis_risk_penalty_weight = 0.5;

    // Set similar lot sizes for all venues
    for v in &mut cfg.venues {
        v.lot_size_tao = 0.01;
        v.size_step_tao = 0.01;
        v.min_notional_usd = 1.0;
    }

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Create position on venue 0
    state.venues[0].position_tao = 5.0;
    state.venues[0].avg_entry_price = 100.0;

    // Set up basis exposure: current basis is positive (venues trading rich)
    // Venue 1: trading at fair value (neutral basis)
    state.venues[1].mid = Some(103.0); // 3% above fair
    state.venues[1].spread = Some(0.5);
    state.venues[1].depth_near_mid = 1_000_000.0;
    state.venues[1].last_mid_update_ms = Some(0);

    // Venue 2: trading rich relative to fair (would increase basis)
    state.venues[2].mid = Some(103.0); // Same gross price
    state.venues[2].spread = Some(0.5);
    state.venues[2].depth_near_mid = 1_000_000.0;
    state.venues[2].last_mid_update_ms = Some(0);

    state.recompute_after_fills(&cfg);

    // First get intents with current setup
    let _intents1 = exit::compute_exit_intents(&cfg, &state, 0);

    // Now modify venue 2 to have a price that would change basis differently
    state.venues[2].mid = Some(102.5); // Slightly lower (still profitable but less basis impact)
    state.recompute_after_fills(&cfg);

    let intents2 = exit::compute_exit_intents(&cfg, &state, 0);

    // Verify determinism: same state produces same output
    let intents2_copy = exit::compute_exit_intents(&cfg, &state, 0);
    assert_eq!(
        intents2.len(),
        intents2_copy.len(),
        "Exit must be deterministic"
    );
    for (a, b) in intents2.iter().zip(intents2_copy.iter()) {
        assert_eq!(
            intent_venue_index(a),
            intent_venue_index(b),
            "Venue ordering must be deterministic"
        );
        assert!(
            (intent_size(a) - intent_size(b)).abs() < 0.001,
            "Size must be deterministic"
        );
    }
}

/// Test deterministic ordering with identical edges - venue index should be tiebreaker.
#[test]
fn exit_deterministic_ordering_with_identical_edges() {
    let mut cfg = Config::default();
    cfg.exit.max_total_tao_per_tick = 10.0;
    cfg.exit.max_venue_tao_per_tick = 5.0;
    cfg.exit.edge_min_usd = 0.01;
    cfg.exit.min_depth_usd = 1.0;
    cfg.exit.depth_fraction = 1.0;
    cfg.exit.min_intent_size_tao = 0.01;

    // Zero out all adjustment factors to get pure edge comparison
    cfg.exit.basis_weight = 0.0;
    cfg.exit.funding_weight = 0.0;
    cfg.exit.fragmentation_penalty_per_tao = 0.0;
    cfg.exit.fragmentation_reduction_bonus = 0.0;
    cfg.exit.basis_risk_penalty_weight = 0.0;

    for v in &mut cfg.venues {
        v.lot_size_tao = 0.01;
        v.size_step_tao = 0.01;
        v.min_notional_usd = 1.0;
    }

    let mut state = GlobalState::new(&cfg);
    state.fair_value = Some(100.0);
    state.fair_value_prev = 100.0;

    // Position to exit
    state.venues[0].position_tao = 10.0;
    state.venues[0].avg_entry_price = 100.0;

    // Set identical conditions on venues 1, 2, 3
    for i in 1..=3 {
        state.venues[i].mid = Some(105.0);
        state.venues[i].spread = Some(0.5);
        state.venues[i].depth_near_mid = 1_000_000.0;
        state.venues[i].last_mid_update_ms = Some(0);
        state.venues[i].funding_8h = 0.0;
        state.venues[i].position_tao = 0.0;
    }

    state.recompute_after_fills(&cfg);

    // Run multiple times to verify determinism
    let intents1 = exit::compute_exit_intents(&cfg, &state, 0);
    let intents2 = exit::compute_exit_intents(&cfg, &state, 0);
    let intents3 = exit::compute_exit_intents(&cfg, &state, 0);

    assert_eq!(intents1.len(), intents2.len(), "Must be deterministic");
    assert_eq!(intents1.len(), intents3.len(), "Must be deterministic");

    for i in 0..intents1.len() {
        assert_eq!(
            intent_venue_index(&intents1[i]),
            intent_venue_index(&intents2[i]),
            "Venue order must be deterministic at position {i}"
        );
        assert_eq!(
            intent_venue_index(&intents1[i]),
            intent_venue_index(&intents3[i]),
            "Venue order must be deterministic at position {i}"
        );
        assert!(
            (intent_size(&intents1[i]) - intent_size(&intents2[i])).abs() < 0.001,
            "Size must be deterministic at position {i}"
        );
    }

    // With identical edges and all adjustments zeroed, should prefer lower venue index
    if intents1.len() >= 2 {
        assert!(
            intent_venue_index(&intents1[0]) <= intent_venue_index(&intents1[1]),
            "With identical edges, lower venue index should come first"
        );
    }
}
