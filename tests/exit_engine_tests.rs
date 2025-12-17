use paraphina::config::Config;
use paraphina::exit;
use paraphina::state::GlobalState;
use paraphina::types::{Side, VenueStatus};

#[test]
fn exit_noops_without_fair_value() {
    let cfg = Config::default();
    let state = GlobalState::new(&cfg);

    let intents = exit::compute_exit_intents(&cfg, &state, 0);
    assert!(intents.is_empty());
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
        assert_eq!(it.side, Side::Sell);
    }

    let total: f64 = intents.iter().map(|i| i.size).sum();
    assert!(
        (total - 10.0).abs() < 1e-6 || total <= 10.0,
        "total exit should be <= max_total_tao_per_tick"
    );

    // With per-venue cap 6, it should split roughly 6 + 4
    assert!(intents.len() >= 2, "should allocate across multiple venues");
    assert!(intents.iter().any(|i| (i.size - 6.0).abs() < 1e-6));
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
        assert!(it.venue_index != 1);
        assert!(it.venue_index != 2);
        assert!(it.venue_index != 3);
    }
}
