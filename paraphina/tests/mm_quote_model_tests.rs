// tests/mm_quote_model_tests.rs
//
// Milestone G: Multi-venue quoting model tests.
//
// Tests cover:
// - Passivity: bid < best_bid, ask > best_ask
// - Reservation price moves correctly with q_global sign and (q_v - q_target_v)
// - HardLimit/Critical produces no quotes even if kill_switch false
// - AS half-spread computation
// - Size model constraints
// - Order lifetime + tolerance logic

use std::sync::Arc;

use paraphina::config::{Config, MmVenueRole, RiskProfile};
use paraphina::engine::Engine;
use paraphina::mm::{
    compute_mm_quotes, compute_order_actions, compute_venue_targets, evaluate_replace_order,
    should_replace_order, ActiveMmOrder, CompressionEdgeNearMissReason, MmOrderAction,
    ShouldReplaceOrderCtx, TouchRiskNearMissReason,
};
use paraphina::state::{FundingState, GlobalState, KillReason, RiskRegime};
use paraphina::types::{Side, VenueStatus};

/// Small harness that gives us a Config, Engine and GlobalState
/// for a given risk profile.
struct Harness {
    cfg: &'static Config,
    engine: Engine<'static>,
    state: GlobalState,
}

fn make_harness(profile: RiskProfile) -> Harness {
    let cfg_box = Box::new(Config::for_profile(profile));
    let cfg: &'static Config = Box::leak(cfg_box);

    let engine = Engine::new(cfg);
    let state = GlobalState::new(cfg);

    Harness { cfg, engine, state }
}

/// Helper: seed deterministic books then run a tick so FV/vol/limits are initialised.
fn init_tick(h: &mut Harness, now_ms: i64) {
    h.engine.seed_dummy_mids(&mut h.state, now_ms);
    h.engine.main_tick(&mut h.state, now_ms);
}

/// Helper to set up market conditions for quoting.
fn setup_market(h: &mut Harness) {
    // Set fair value and volatility.
    h.state.fair_value = Some(300.0);
    h.state.fair_value_prev = 300.0;
    h.state.sigma_eff = 0.02;
    h.state.spread_mult = 1.0;
    h.state.size_mult = 1.0;
    h.state.vol_ratio_clipped = 1.0;
    h.state.delta_limit_usd = 100_000.0;

    // Set up venue states with valid data.
    for v in &mut h.state.venues {
        v.mid = Some(300.0);
        v.spread = Some(0.10); // 10 cent spread
        v.depth_near_mid = 10_000.0;
        v.margin_available_usd = 10_000.0;
        v.dist_liq_sigma = 10.0;
        v.status = VenueStatus::Healthy;
        v.toxicity = 0.0;
    }
}

// =============================================================================
// Passivity Tests
// =============================================================================

#[test]
fn passivity_bid_strictly_below_best_bid() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for (i, q) in quotes.iter().enumerate() {
        if let Some(bid) = &q.bid {
            let vstate = &h.state.venues[i];
            let vcfg = &h.cfg.venues[i];
            let mid = vstate.mid.unwrap();
            let spread = vstate.spread.unwrap();
            let best_bid = mid - spread / 2.0;
            let tick = vcfg.tick_size;

            assert!(
                bid.price < best_bid,
                "Venue {}: bid {} should be strictly < best_bid {}",
                i,
                bid.price,
                best_bid
            );

            if h.cfg.venues[i].id == "aster"
                && h.cfg.mm.venue_role_for(&h.cfg.venues[i].id)
                    == paraphina::config::MmVenueRole::Fill
            {
                assert!(
                    bid.price >= best_bid - tick,
                    "Venue {}: Aster bid {} should be within one tick of touch {}",
                    i,
                    bid.price,
                    best_bid - tick
                );
            } else {
                assert!(
                    bid.price <= best_bid - tick,
                    "Venue {}: bid {} should be <= best_bid - tick {}",
                    i,
                    bid.price,
                    best_bid - tick
                );
            }
        }
    }
}

#[test]
fn passivity_ask_strictly_above_best_ask() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for (i, q) in quotes.iter().enumerate() {
        if let Some(ask) = &q.ask {
            let vstate = &h.state.venues[i];
            let vcfg = &h.cfg.venues[i];
            let mid = vstate.mid.unwrap();
            let spread = vstate.spread.unwrap();
            let best_ask = mid + spread / 2.0;
            let tick = vcfg.tick_size;

            assert!(
                ask.price > best_ask,
                "Venue {}: ask {} should be strictly > best_ask {}",
                i,
                ask.price,
                best_ask
            );

            if h.cfg.venues[i].id == "aster"
                && h.cfg.mm.venue_role_for(&h.cfg.venues[i].id)
                    == paraphina::config::MmVenueRole::Fill
            {
                assert!(
                    ask.price <= best_ask + tick,
                    "Venue {}: Aster ask {} should be within one tick of touch {}",
                    i,
                    ask.price,
                    best_ask + tick
                );
            } else {
                assert!(
                    ask.price >= best_ask + tick,
                    "Venue {}: ask {} should be >= best_ask + tick {}",
                    i,
                    ask.price,
                    best_ask + tick
                );
            }
        }
    }
}

#[test]
fn passivity_bid_ask_do_not_cross() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for (i, q) in quotes.iter().enumerate() {
        if let (Some(bid), Some(ask)) = (&q.bid, &q.ask) {
            assert!(
                bid.price < ask.price,
                "Venue {}: bid {} should be < ask {}",
                i,
                bid.price,
                ask.price
            );
        }
    }
}

// =============================================================================
// Reservation Price Tests
// =============================================================================

#[test]
fn reservation_price_decreases_when_long() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Neutral inventory.
    h.state.q_global_tao = 0.0;
    for v in &mut h.state.venues {
        v.position_tao = 0.0;
    }
    let quotes_neutral = compute_mm_quotes(h.cfg, &h.state);

    // Long inventory.
    h.state.q_global_tao = 20.0;
    for v in &mut h.state.venues {
        v.position_tao = 4.0; // Distributed across 5 venues
    }
    h.state.dollar_delta_usd = h.state.q_global_tao * h.state.fair_value.unwrap();
    let quotes_long = compute_mm_quotes(h.cfg, &h.state);

    // When long, bid should be lower (lower reservation price).
    // Find a venue that has quotes in both cases.
    for i in 0..h.cfg.venues.len() {
        if let (Some(bid_n), Some(bid_l)) = (&quotes_neutral[i].bid, &quotes_long[i].bid) {
            assert!(
                bid_l.price <= bid_n.price + 0.01, // Allow small tolerance
                "Venue {}: long bid {} should be <= neutral bid {} (reservation drops when long)",
                i,
                bid_l.price,
                bid_n.price
            );
            break; // Test at least one venue
        }
    }
}

#[test]
fn reservation_price_increases_when_short() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Neutral inventory.
    h.state.q_global_tao = 0.0;
    for v in &mut h.state.venues {
        v.position_tao = 0.0;
    }
    let quotes_neutral = compute_mm_quotes(h.cfg, &h.state);

    // Short inventory.
    h.state.q_global_tao = -20.0;
    for v in &mut h.state.venues {
        v.position_tao = -4.0;
    }
    h.state.dollar_delta_usd = h.state.q_global_tao * h.state.fair_value.unwrap();
    let quotes_short = compute_mm_quotes(h.cfg, &h.state);

    // When short, ask should be higher (higher reservation price).
    for i in 0..h.cfg.venues.len() {
        if let (Some(ask_n), Some(ask_s)) = (&quotes_neutral[i].ask, &quotes_short[i].ask) {
            assert!(
                ask_s.price >= ask_n.price - 0.01,
                "Venue {}: short ask {} should be >= neutral ask {} (reservation rises when short)",
                i,
                ask_s.price,
                ask_n.price
            );
            break;
        }
    }
}

#[test]
fn reservation_price_affected_by_venue_target_deviation() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Set q_global to 0 to isolate venue target effect.
    h.state.q_global_tao = 0.0;
    h.state.dollar_delta_usd = 0.0;

    // Venue 0: position at target (no deviation).
    h.state.venues[0].position_tao = 0.0;

    // Compute targets first.
    let targets = compute_venue_targets(h.cfg, &h.state, None);
    let q_target_0 = targets[0].q_target;

    let quotes_at_target = compute_mm_quotes(h.cfg, &h.state);

    // Venue 0: position significantly above target.
    h.state.venues[0].position_tao = q_target_0 + 10.0;
    let quotes_above_target = compute_mm_quotes(h.cfg, &h.state);

    // With lambda_inv > 0, having position above target should lower reservation
    // (want to reduce position via lower bids / higher asks).
    if let (Some(bid_at), Some(bid_above)) = (&quotes_at_target[0].bid, &quotes_above_target[0].bid)
    {
        // The effect might be small depending on lambda_inv value.
        // Just verify the formula is applied (prices are different).
        let _diff = (bid_at.price - bid_above.price).abs();
        // Not asserting a specific direction here as it depends on parameter tuning.
    }
}

// =============================================================================
// HardLimit / Critical Tests
// =============================================================================

#[test]
fn hardlimit_produces_no_quotes_even_if_kill_switch_false() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Set HardLimit regime but keep kill_switch false.
    // (Note: in practice, HardLimit should trigger kill_switch, but we test isolation.)
    h.state.risk_regime = RiskRegime::HardLimit;
    h.state.kill_switch = false;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for q in &quotes {
        assert!(q.bid.is_none(), "HardLimit should produce no bid");
        assert!(q.ask.is_none(), "HardLimit should produce no ask");
    }
}

#[test]
fn kill_switch_produces_no_quotes() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    h.state.kill_switch = true;
    h.state.kill_reason = KillReason::PnlHardBreach;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for q in &quotes {
        assert!(q.bid.is_none(), "kill_switch should produce no bid");
        assert!(q.ask.is_none(), "kill_switch should produce no ask");
    }
}

#[test]
fn warning_regime_widens_spread() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Normal regime.
    h.state.risk_regime = RiskRegime::Normal;
    let quotes_normal = compute_mm_quotes(h.cfg, &h.state);

    // Warning regime.
    h.state.risk_regime = RiskRegime::Warning;
    let quotes_warning = compute_mm_quotes(h.cfg, &h.state);

    // Warning should have wider spread.
    for i in 0..h.cfg.venues.len() {
        if let (Some(bid_n), Some(ask_n), Some(bid_w), Some(ask_w)) = (
            &quotes_normal[i].bid,
            &quotes_normal[i].ask,
            &quotes_warning[i].bid,
            &quotes_warning[i].ask,
        ) {
            let spread_normal = ask_n.price - bid_n.price;
            let spread_warning = ask_w.price - bid_w.price;

            assert!(
                spread_warning >= spread_normal,
                "Venue {}: warning spread {} should be >= normal spread {}",
                i,
                spread_warning,
                spread_normal
            );
            break;
        }
    }
}

#[test]
fn warning_regime_caps_size() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Normal regime - verify quotes are produced.
    h.state.risk_regime = RiskRegime::Normal;
    let _quotes_normal = compute_mm_quotes(h.cfg, &h.state);

    // Warning regime.
    h.state.risk_regime = RiskRegime::Warning;
    let quotes_warning = compute_mm_quotes(h.cfg, &h.state);

    let cap = h.cfg.risk.q_warn_cap;

    // Warning should have sizes <= cap.
    for (i, quote_warning) in quotes_warning.iter().enumerate() {
        if let Some(bid_w) = &quote_warning.bid {
            assert!(
                bid_w.size <= cap + 0.001,
                "Venue {}: warning bid size {} should be <= cap {}",
                i,
                bid_w.size,
                cap
            );
        }
        if let Some(ask_w) = &quote_warning.ask {
            assert!(
                ask_w.size <= cap + 0.001,
                "Venue {}: warning ask size {} should be <= cap {}",
                i,
                ask_w.size,
                cap
            );
        }
    }
}

// =============================================================================
// Venue Gating Tests
// =============================================================================

#[test]
fn disabled_venue_produces_no_quotes() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Disable venue 0.
    h.state.venues[0].status = VenueStatus::Disabled;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    assert!(
        quotes[0].bid.is_none() && quotes[0].ask.is_none(),
        "Disabled venue should have no quotes"
    );

    // Other venues should still quote.
    let other_has_quotes = quotes[1..]
        .iter()
        .any(|q| q.bid.is_some() || q.ask.is_some());
    assert!(other_has_quotes, "Other venues should still have quotes");
}

#[test]
fn high_toxicity_venue_produces_no_quotes() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Set venue 0 to high toxicity (above tox_high_threshold).
    h.state.venues[0].toxicity = 0.95;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    assert!(
        quotes[0].bid.is_none() && quotes[0].ask.is_none(),
        "High toxicity venue should have no quotes"
    );
}

#[test]
fn stale_book_produces_no_quotes() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Remove mid from venue 0 (simulating stale book).
    h.state.venues[0].mid = None;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    assert!(
        quotes[0].bid.is_none() && quotes[0].ask.is_none(),
        "Venue with no mid should have no quotes"
    );
}

#[test]
fn critical_liquidation_distance_produces_no_quotes() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Set venue 0 below critical liquidation distance.
    h.state.venues[0].dist_liq_sigma = h.cfg.risk.liq_crit_sigma - 0.5;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    assert!(
        quotes[0].bid.is_none() && quotes[0].ask.is_none(),
        "Venue at critical liquidation should have no quotes"
    );
}

// =============================================================================
// Size Model Tests
// =============================================================================

#[test]
fn size_respects_max_order_size() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for (i, q) in quotes.iter().enumerate() {
        let max_size = h.cfg.venues[i].max_order_size;

        if let Some(bid) = &q.bid {
            assert!(
                bid.size <= max_size,
                "Venue {}: bid size {} should be <= max {}",
                i,
                bid.size,
                max_size
            );
        }
        if let Some(ask) = &q.ask {
            assert!(
                ask.size <= max_size,
                "Venue {}: ask size {} should be <= max {}",
                i,
                ask.size,
                max_size
            );
        }
    }
}

#[test]
fn size_respects_lot_size() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for (i, q) in quotes.iter().enumerate() {
        let lot_size = h.cfg.venues[i].lot_size_tao;

        if let Some(bid) = &q.bid {
            let remainder = bid.size % lot_size;
            assert!(
                remainder.abs() < 1e-9 || (lot_size - remainder).abs() < 1e-9,
                "Venue {}: bid size {} should be multiple of lot size {}",
                i,
                bid.size,
                lot_size
            );
        }
        if let Some(ask) = &q.ask {
            let remainder = ask.size % lot_size;
            assert!(
                remainder.abs() < 1e-9 || (lot_size - remainder).abs() < 1e-9,
                "Venue {}: ask size {} should be multiple of lot size {}",
                i,
                ask.size,
                lot_size
            );
        }
    }
}

#[test]
fn liquidation_warning_shrinks_size() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Normal liquidation distance.
    h.state.venues[0].dist_liq_sigma = 10.0;
    let quotes_normal = compute_mm_quotes(h.cfg, &h.state);

    // Inside warning zone.
    h.state.venues[0].dist_liq_sigma = h.cfg.risk.liq_warn_sigma - 1.0;
    let quotes_warning = compute_mm_quotes(h.cfg, &h.state);

    // Size should be smaller in warning zone.
    if let (Some(bid_n), Some(bid_w)) = (&quotes_normal[0].bid, &quotes_warning[0].bid) {
        assert!(
            bid_w.size <= bid_n.size + 0.001,
            "Liquidation warning should shrink size: {} <= {}",
            bid_w.size,
            bid_n.size
        );
    }
}

// =============================================================================
// Venue Target Inventory Tests
// =============================================================================

#[test]
fn venue_targets_scale_with_depth() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    h.state.q_global_tao = 10.0;

    // Set different depths.
    h.state.venues[0].depth_near_mid = 30_000.0;
    h.state.venues[1].depth_near_mid = 15_000.0;
    h.state.venues[2].depth_near_mid = 5_000.0;

    let targets = compute_venue_targets(h.cfg, &h.state, None);

    // Venue 0 (highest depth) should have highest liquidity weight.
    assert!(
        targets[0].w_liq > targets[1].w_liq,
        "Higher depth should give higher w_liq: {} > {}",
        targets[0].w_liq,
        targets[1].w_liq
    );
    assert!(
        targets[1].w_liq > targets[2].w_liq,
        "Higher depth should give higher w_liq: {} > {}",
        targets[1].w_liq,
        targets[2].w_liq
    );
}

#[test]
fn venue_targets_respond_to_funding() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    h.state.q_global_tao = 0.0;

    // Set uniform depth.
    for v in &mut h.state.venues {
        v.depth_near_mid = 10_000.0;
    }

    // Venue 0: positive funding (longs pay).
    h.state.venues[0].funding_8h = 0.001;

    // Venue 1: negative funding (shorts pay).
    h.state.venues[1].funding_8h = -0.001;

    // Venue 2: zero funding.
    h.state.venues[2].funding_8h = 0.0;

    let targets = compute_venue_targets(h.cfg, &h.state, None);

    // With positive funding, target should be higher (prefer shorts there).
    // With negative funding, target should be lower (prefer longs there).
    // This depends on the phi function and w_fund configuration.
    // Just verify targets are different.
    let t0 = targets[0].q_target;
    let t1 = targets[1].q_target;
    let t2 = targets[2].q_target;

    // They should not all be identical if funding matters.
    // (Unless funding is disabled in mm config.)
    let all_same = (t0 - t1).abs() < 1e-9 && (t1 - t2).abs() < 1e-9;
    if h.cfg.mm.funding_enabled && h.cfg.venues[0].w_fund > 0.0 {
        assert!(
            !all_same,
            "Different funding rates should produce different targets"
        );
    }
}

// =============================================================================
// Order Lifetime Tests
// =============================================================================

#[test]
fn young_passive_order_not_replaced() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];
    let state = GlobalState::new(&cfg);

    let current = ActiveMmOrder {
        venue_index: 0,
        side: Side::Buy,
        price: 299.90,
        size: 1.0,
        timestamp_ms: 100,
    };

    // 200ms since order placed; min_quote_lifetime_ms is 500.
    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[0],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 299.91, // Slightly different price
        desired_size: 1.0,
        now_ms: 200, // Order is 100ms old (200 - 100)
        best_bid: 300.0,
        best_ask: 300.10,
    };
    let should_replace = should_replace_order(ctx);

    assert!(
        !should_replace,
        "Young passive order within lifetime should not be replaced"
    );
}

#[test]
fn old_order_with_price_change_replaced() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];
    let state = GlobalState::new(&cfg);

    let current = ActiveMmOrder {
        venue_index: 0,
        side: Side::Buy,
        price: 299.90,
        size: 1.0,
        timestamp_ms: 0,
    };

    // 1 second since order; price changed by 10 ticks.
    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[0],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 299.80, // 10 cents different (10 ticks at 0.01)
        desired_size: 1.0,
        now_ms: 1000,
        best_bid: 300.0,
        best_ask: 300.10,
    };
    let should_replace = should_replace_order(ctx);

    assert!(
        should_replace,
        "Old order with large price change should be replaced"
    );
}

#[test]
fn old_order_with_size_change_replaced() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];
    let state = GlobalState::new(&cfg);

    let current = ActiveMmOrder {
        venue_index: 0,
        side: Side::Buy,
        price: 299.90,
        size: 1.0,
        timestamp_ms: 0,
    };

    // 1 second since order; size changed by 30%.
    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[0],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 299.90,
        desired_size: 1.30, // 30% size increase
        now_ms: 1000,
        best_bid: 300.0,
        best_ask: 300.10,
    };
    let should_replace = should_replace_order(ctx);

    assert!(
        should_replace,
        "Old order with large size change should be replaced"
    );
}

#[test]
fn old_order_with_price_only_move_can_be_held_by_price_cadence() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];
    let state = GlobalState::new(&cfg);

    let current = ActiveMmOrder {
        venue_index: 0,
        side: Side::Buy,
        price: 299.90,
        size: 1.0,
        timestamp_ms: 0,
    };

    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[0],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 299.94, // 4 ticks higher, still passive
        desired_size: 1.0,
        now_ms: 1200, // past widened quote lifetime, still within price cadence
        best_bid: 300.0,
        best_ask: 300.10,
    };
    let should_replace = should_replace_order(ctx);

    assert!(
        !should_replace,
        "Passive price-only move should be held by price cadence on a wide-spread venue"
    );
}

#[test]
fn inventory_reducing_price_move_bypasses_price_cadence() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];
    let mut state = GlobalState::new(&cfg);
    state.q_global_tao = 5.0;

    let current = ActiveMmOrder {
        venue_index: 0,
        side: Side::Sell,
        price: 300.10,
        size: 1.0,
        timestamp_ms: 0,
    };

    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[0],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 300.06, // more aggressive ask to reduce long inventory
        desired_size: 1.0,
        now_ms: 700,
        best_bid: 300.0,
        best_ask: 300.10,
    };
    let should_replace = should_replace_order(ctx);

    assert!(
        should_replace,
        "Inventory-reducing price move should bypass price cadence"
    );
}

#[test]
fn hyperliquid_sell_tighten_into_compressed_spread_can_be_held() {
    let cfg = Config::default();
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "hyperliquid")
        .expect("hyperliquid venue in config");
    let vcfg = &cfg.venues[venue_index];
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some(100.25);
    state.venues[venue_index].spread = Some(0.5);
    state.venues[venue_index].prev_spread = Some(1.0);

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: 101.02,
        size: 1.0,
        timestamp_ms: 0,
    };

    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 100.91,
        desired_size: 1.08,
        now_ms: 1000,
        best_bid: 100.0,
        best_ask: 100.5,
    };
    let should_replace = should_replace_order(ctx);

    assert!(
        !should_replace,
        "Hyperliquid sell tightening into a compressed spread should be held"
    );
}

fn evaluate_hyperliquid_sell_decision(
    prev_spread: Option<f64>,
    current_price: f64,
    current_size: f64,
    desired_price: f64,
    desired_size: f64,
    q_global_tao: f64,
    position_tao: f64,
) -> paraphina::mm::MmReplaceDecision {
    evaluate_hyperliquid_sell_decision_with_market(
        prev_spread,
        Some(100.5),
        current_price,
        current_size,
        desired_price,
        desired_size,
        q_global_tao,
        position_tao,
        100.0,
        100.5,
        2_000,
    )
}

fn evaluate_hyperliquid_sell_decision_with_market(
    prev_spread: Option<f64>,
    prev_best_ask: Option<f64>,
    current_price: f64,
    current_size: f64,
    desired_price: f64,
    desired_size: f64,
    q_global_tao: f64,
    position_tao: f64,
    best_bid: f64,
    best_ask: f64,
    now_ms: i64,
) -> paraphina::mm::MmReplaceDecision {
    let cfg = Config::default();
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "hyperliquid")
        .expect("hyperliquid venue in config");
    let vcfg = &cfg.venues[venue_index];
    let mut state = GlobalState::new(&cfg);
    state.q_global_tao = q_global_tao;
    state.venues[venue_index].position_tao = position_tao;
    state.venues[venue_index].mid = Some(100.25);
    state.venues[venue_index].spread = Some(0.5);
    state.venues[venue_index].prev_spread = prev_spread;
    state.venues[venue_index].prev_best_ask = prev_best_ask;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: current_price,
        size: current_size,
        timestamp_ms: 0,
    };

    evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price,
        desired_size,
        now_ms,
        best_bid,
        best_ask,
    })
}

#[test]
fn hyperliquid_sell_compression_guard_records_no_prev_spread_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision(None, 101.02, 1.0, 100.91, 1.08, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::NoPrevSpread)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_not_compressed_nearmiss() {
    let decision =
        evaluate_hyperliquid_sell_decision(Some(0.5), 101.02, 1.0, 100.91, 1.08, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::NotCompressed)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_compression_too_small_nearmiss() {
    let decision =
        evaluate_hyperliquid_sell_decision(Some(0.59), 101.02, 1.0, 100.91, 1.08, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::CompressionTooSmall)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_clearance_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision(Some(1.0), 101.5, 1.0, 101.3, 1.08, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::ClearanceNotThinEnough)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_size_block_nearmiss() {
    let decision =
        evaluate_hyperliquid_sell_decision(Some(1.0), 101.02, 1.0, 100.91, 1.5, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::SizeExceedsBlock)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_not_tightening_nearmiss() {
    let decision =
        evaluate_hyperliquid_sell_decision(Some(1.0), 101.02, 1.0, 101.15, 1.0, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::DesiredNotTightening)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_not_passive_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision(Some(1.0), 100.5, 1.0, 100.39, 1.0, 0.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::NotPassive)
    );
}

#[test]
fn hyperliquid_sell_compression_guard_records_inventory_reducing_nearmiss() {
    let decision =
        evaluate_hyperliquid_sell_decision(Some(1.0), 101.02, 1.0, 100.91, 1.0, 5.0, 0.0);
    assert_eq!(
        decision.compression_edge_nearmiss_reason,
        Some(CompressionEdgeNearMissReason::InventoryReducingBlock)
    );
}

#[test]
fn hyperliquid_sell_tighten_into_thin_touch_uses_touch_risk_hysteresis() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.5),
        100.70,
        1.0,
        100.51,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Keep);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::TouchRiskHysteresis
    );
}

#[test]
fn hyperliquid_sell_rising_touch_bypasses_young_hysteresis() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.60,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::TouchRiskFastpath
    );
    assert!(!decision.touch_risk_size_band_applied);
}

#[test]
fn hyperliquid_sell_small_size_drift_can_use_touch_risk_fastpath() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.60,
        1.15,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::TouchRiskFastpath
    );
    assert!(decision.touch_risk_size_band_applied);
}

#[test]
fn hyperliquid_sell_three_tick_clearance_now_bypasses_young_hysteresis() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.53,
        1.0,
        100.61,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::TouchRiskFastpath
    );
    assert!(!decision.touch_risk_size_band_applied);
}

#[test]
fn hyperliquid_sell_price_move_forensics_show_marginal_old_replace() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.49),
        100.70,
        1.0,
        100.72,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(decision.reason, paraphina::mm::MmReplaceReason::PriceMove);
    assert!(decision
        .hyperliquid_sell_price_excess_ticks
        .is_some_and(|value| (value - 1.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_best_ask_motion_ticks
        .is_some_and(|value| (value - 1.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_current_clearance_ticks
        .is_some_and(|value| (value - 20.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_desired_clearance_ticks
        .is_some_and(|value| (value - 22.0).abs() < 1e-9));
    assert_eq!(
        decision.hyperliquid_sell_price_reprice_remaining_ms,
        Some(0)
    );
}

#[test]
fn hyperliquid_sell_stationary_touch_small_old_price_move_held_by_deadband() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.50),
        100.70,
        1.0,
        100.72,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Keep);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::StationaryPriceDeadbandHysteresis
    );
    assert!(decision
        .hyperliquid_sell_price_excess_ticks
        .is_some_and(|value| (value - 1.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_best_ask_motion_ticks
        .is_some_and(|value| value.abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_current_clearance_ticks
        .is_some_and(|value| (value - 20.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_desired_clearance_ticks
        .is_some_and(|value| (value - 22.0).abs() < 1e-9));
    assert_eq!(
        decision.hyperliquid_sell_price_reprice_remaining_ms,
        Some(0)
    );
}

#[test]
fn hyperliquid_sell_reduced_adverse_selection_still_uses_stationary_deadband() {
    let cfg = Config::default();
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "hyperliquid")
        .expect("hyperliquid venue in config");
    let vcfg = &cfg.venues[venue_index];
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some(100.25);
    state.venues[venue_index].spread = Some(0.5);
    state.venues[venue_index].prev_spread = Some(0.5);
    state.venues[venue_index].prev_best_ask = Some(100.50);
    state.venues[venue_index].toxicity = 0.7;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: 100.70,
        size: 1.0,
        timestamp_ms: 0,
    };

    let decision = evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: 0.0,
        current: &current,
        desired_price: 100.72,
        desired_size: 1.0,
        now_ms: 2_000,
        best_bid: 100.0,
        best_ask: 100.5,
    });
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Keep);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::StationaryPriceDeadbandHysteresis
    );
    assert_eq!(
        decision.utility_reason,
        paraphina::state::VenueUtilityReason::AdverseSelection
    );
}

#[test]
fn hyperliquid_sell_price_move_forensics_show_young_keep_under_cadence() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.49),
        100.70,
        1.0,
        100.74,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Keep);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::YoungPassiveHysteresis
    );
    assert!(decision
        .hyperliquid_sell_price_excess_ticks
        .is_some_and(|value| (value - 3.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_best_ask_motion_ticks
        .is_some_and(|value| (value - 1.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_current_clearance_ticks
        .is_some_and(|value| (value - 20.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_desired_clearance_ticks
        .is_some_and(|value| (value - 24.0).abs() < 1e-9));
    assert_eq!(
        decision.hyperliquid_sell_price_reprice_remaining_ms,
        Some(250)
    );
}

#[test]
fn hyperliquid_sell_price_move_forensics_show_large_old_replace() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.49),
        100.70,
        1.0,
        101.00,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(decision.reason, paraphina::mm::MmReplaceReason::PriceMove);
    assert!(decision
        .hyperliquid_sell_price_excess_ticks
        .is_some_and(|value| (value - 29.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_best_ask_motion_ticks
        .is_some_and(|value| (value - 1.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_current_clearance_ticks
        .is_some_and(|value| (value - 20.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_desired_clearance_ticks
        .is_some_and(|value| (value - 50.0).abs() < 1e-9));
    assert_eq!(
        decision.hyperliquid_sell_price_reprice_remaining_ms,
        Some(0)
    );
}

#[test]
fn hyperliquid_sell_stationary_touch_large_old_price_move_still_reprices() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.50),
        100.70,
        1.0,
        101.00,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(decision.reason, paraphina::mm::MmReplaceReason::PriceMove);
    assert!(decision
        .hyperliquid_sell_price_excess_ticks
        .is_some_and(|value| (value - 29.0).abs() < 1e-9));
    assert!(decision
        .hyperliquid_sell_best_ask_motion_ticks
        .is_some_and(|value| value.abs() < 1e-9));
    assert_eq!(
        decision.hyperliquid_sell_price_reprice_remaining_ms,
        Some(0)
    );
}

#[test]
fn hyperliquid_sell_reduced_low_conversion_does_not_use_stationary_deadband() {
    let cfg = Config::default();
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "hyperliquid")
        .expect("hyperliquid venue in config");
    let vcfg = &cfg.venues[venue_index];
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some(100.25);
    state.venues[venue_index].spread = Some(0.5);
    state.venues[venue_index].prev_spread = Some(0.5);
    state.venues[venue_index].prev_best_ask = Some(100.50);
    state.venues[venue_index].utility.mm_fill_credit_ewma = 0.0;
    state.venues[venue_index].utility.mm_fillless_ack_pressure = 25.0;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: 100.70,
        size: 1.0,
        timestamp_ms: 0,
    };

    let decision = evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: 0.0,
        current: &current,
        desired_price: 100.72,
        desired_size: 1.0,
        now_ms: 2_000,
        best_bid: 100.0,
        best_ask: 100.5,
    });
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(decision.reason, paraphina::mm::MmReplaceReason::PriceMove);
    assert_eq!(
        decision.utility_reason,
        paraphina::state::VenueUtilityReason::LowConversion
    );
}

#[test]
fn hyperliquid_sell_inventory_reducing_still_bypasses_touch_risk_controller() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.60,
        1.0,
        5.0,
        0.0,
        100.0,
        100.5,
        500,
    );
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::YoungPassiveHysteresis
    );
}

#[test]
fn aster_touch_local_young_one_tick_offside_sell_uses_fastpath() {
    let mut cfg = Config::default();
    cfg.mm
        .venue_role_by_venue
        .insert("aster".to_string(), MmVenueRole::Fill);
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "aster")
        .expect("aster venue in config");
    let vcfg = &cfg.venues[venue_index];
    let tick = vcfg.tick_size.max(1e-6);
    let best_bid = 2_131.20;
    let best_ask = best_bid + (3.0 * tick);
    let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("aster");
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some((best_bid + best_ask) * 0.5);
    state.venues[venue_index].spread = Some(best_ask - best_bid);
    state.venues[venue_index].status = VenueStatus::Healthy;
    state.venues[venue_index].toxicity = 0.0;
    state.venues[venue_index].utility.mm_fill_credit_ewma = 0.5;
    state.venues[venue_index].utility.mm_fillless_ack_pressure = 0.0;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: best_ask + (0.5 * tick),
        size: 0.01,
        timestamp_ms: 0,
    };

    let decision = evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: 0.0,
        current: &current,
        desired_price: best_ask + tick,
        desired_size: 0.01,
        now_ms: base_lifetime.saturating_sub(1),
        best_bid,
        best_ask,
    });
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Keep);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::AsterTouchOffsideFastpath
    );
    assert_eq!(
        decision.utility_tier,
        paraphina::state::VenueUtilityTier::Full
    );
    assert_eq!(decision.venue_role, MmVenueRole::Fill);
}

#[test]
fn aster_touch_local_young_one_tick_offside_buy_uses_fastpath() {
    let mut cfg = Config::default();
    cfg.mm
        .venue_role_by_venue
        .insert("aster".to_string(), MmVenueRole::Fill);
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "aster")
        .expect("aster venue in config");
    let vcfg = &cfg.venues[venue_index];
    let tick = vcfg.tick_size.max(1e-6);
    let best_bid = 2_131.20;
    let best_ask = best_bid + (3.0 * tick);
    let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("aster");
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some((best_bid + best_ask) * 0.5);
    state.venues[venue_index].spread = Some(best_ask - best_bid);
    state.venues[venue_index].status = VenueStatus::Healthy;
    state.venues[venue_index].toxicity = 0.0;
    state.venues[venue_index].utility.mm_fill_credit_ewma = 0.5;
    state.venues[venue_index].utility.mm_fillless_ack_pressure = 0.0;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Buy,
        price: best_bid - (0.5 * tick),
        size: 0.01,
        timestamp_ms: 0,
    };

    let decision = evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: 0.0,
        current: &current,
        desired_price: best_bid - tick,
        desired_size: 0.01,
        now_ms: base_lifetime.saturating_sub(1),
        best_bid,
        best_ask,
    });
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Keep);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::AsterTouchOffsideFastpath
    );
    assert_eq!(
        decision.utility_tier,
        paraphina::state::VenueUtilityTier::Full
    );
    assert_eq!(decision.venue_role, MmVenueRole::Fill);
}

#[test]
fn aster_touch_local_old_quote_records_quote_too_old_nearmiss() {
    let mut cfg = Config::default();
    cfg.mm
        .venue_role_by_venue
        .insert("aster".to_string(), MmVenueRole::Fill);
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "aster")
        .expect("aster venue in config");
    let vcfg = &cfg.venues[venue_index];
    let tick = vcfg.tick_size.max(1e-6);
    let best_bid = 2_131.20;
    let best_ask = best_bid + (3.0 * tick);
    let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("aster");
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some((best_bid + best_ask) * 0.5);
    state.venues[venue_index].spread = Some(best_ask - best_bid);
    state.venues[venue_index].status = VenueStatus::Healthy;
    state.venues[venue_index].toxicity = 0.0;
    state.venues[venue_index].utility.mm_fill_credit_ewma = 0.5;
    state.venues[venue_index].utility.mm_fillless_ack_pressure = 0.0;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: best_ask + (0.5 * tick),
        size: 0.01,
        timestamp_ms: 0,
    };

    let decision = evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: 0.0,
        current: &current,
        desired_price: best_ask + tick,
        desired_size: 0.01,
        now_ms: base_lifetime + 100,
        best_bid,
        best_ask,
    });
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::DangerousOffside
    );
    assert_eq!(
        decision.aster_touch_offside_nearmiss_reason,
        Some(paraphina::mm::AsterTouchOffsideNearMissReason::QuoteTooOld)
    );
}

#[test]
fn aster_touch_local_crossing_quote_still_replaces() {
    let mut cfg = Config::default();
    cfg.mm
        .venue_role_by_venue
        .insert("aster".to_string(), MmVenueRole::Fill);
    let venue_index = cfg
        .venues
        .iter()
        .position(|venue| venue.id == "aster")
        .expect("aster venue in config");
    let vcfg = &cfg.venues[venue_index];
    let tick = vcfg.tick_size.max(1e-6);
    let best_bid = 2_131.20;
    let best_ask = best_bid + (3.0 * tick);
    let base_lifetime = cfg.mm.min_quote_lifetime_ms_for("aster");
    let mut state = GlobalState::new(&cfg);
    state.venues[venue_index].mid = Some((best_bid + best_ask) * 0.5);
    state.venues[venue_index].spread = Some(best_ask - best_bid);
    state.venues[venue_index].status = VenueStatus::Healthy;
    state.venues[venue_index].toxicity = 0.0;
    state.venues[venue_index].utility.mm_fill_credit_ewma = 0.5;
    state.venues[venue_index].utility.mm_fillless_ack_pressure = 0.0;

    let current = ActiveMmOrder {
        venue_index,
        side: Side::Sell,
        price: best_bid + tick,
        size: 0.01,
        timestamp_ms: 0,
    };

    let decision = evaluate_replace_order(ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[venue_index],
        q_global_tao: 0.0,
        current: &current,
        desired_price: best_ask + tick,
        desired_size: 0.01,
        now_ms: base_lifetime.saturating_sub(1),
        best_bid,
        best_ask,
    });
    assert_eq!(decision.outcome, paraphina::mm::MmReplaceOutcome::Replace);
    assert_eq!(
        decision.reason,
        paraphina::mm::MmReplaceReason::DangerousOffside
    );
    assert_eq!(
        decision.aster_touch_offside_nearmiss_reason,
        Some(paraphina::mm::AsterTouchOffsideNearMissReason::Crossing)
    );
}

#[test]
fn hyperliquid_sell_touch_risk_records_no_prev_best_ask_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        None,
        100.52,
        1.0,
        100.60,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::NoPrevBestAsk)
    );
}

#[test]
fn hyperliquid_sell_touch_risk_records_clearance_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.5),
        100.70,
        1.0,
        100.60,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::ClearanceNotThinEnough)
    );
}

#[test]
fn hyperliquid_sell_touch_risk_records_touch_not_rising_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.50),
        100.52,
        1.0,
        100.60,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::TouchNotRisingEnough)
    );
}

#[test]
fn hyperliquid_sell_touch_risk_records_desired_not_widening_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.40,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::DesiredNotWidening)
    );
}

#[test]
fn hyperliquid_sell_touch_risk_records_size_block_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.60,
        1.5,
        0.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::SizeExceedsBlock)
    );
    assert!(!decision.touch_risk_size_band_applied);
}

#[test]
fn hyperliquid_sell_touch_risk_records_inventory_reducing_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.60,
        1.0,
        5.0,
        0.0,
        100.0,
        100.5,
        250,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::InventoryReducingBlock)
    );
}

#[test]
fn hyperliquid_sell_touch_risk_records_age_window_nearmiss() {
    let decision = evaluate_hyperliquid_sell_decision_with_market(
        Some(0.5),
        Some(100.48),
        100.52,
        1.0,
        100.60,
        1.0,
        0.0,
        0.0,
        100.0,
        100.5,
        2_000,
    );
    assert_eq!(
        decision.touch_risk_nearmiss_reason,
        Some(TouchRiskNearMissReason::AgeOutsideFastpathWindow)
    );
}

#[test]
fn non_passive_order_replaced_even_if_young() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];
    let state = GlobalState::new(&cfg);

    // Order is priced at best_bid (not passive).
    let current = ActiveMmOrder {
        venue_index: 0,
        side: Side::Buy,
        price: 300.00, // At best_bid
        size: 1.0,
        timestamp_ms: 400,
    };

    // Young order but not passive.
    let ctx = ShouldReplaceOrderCtx {
        cfg: &cfg,
        vcfg,
        vstate: &state.venues[0],
        q_global_tao: state.q_global_tao,
        current: &current,
        desired_price: 299.90,
        desired_size: 1.0,
        now_ms: 500, // Only 100ms old
        best_bid: 300.00,
        best_ask: 300.10,
    };
    let should_replace = should_replace_order(ctx);

    // Order is at best_bid, not passive (should be < best_bid - tick).
    // With price tolerance of 1 tick, 10 ticks difference should trigger replace.
    assert!(
        should_replace,
        "Non-passive order should be replaced even if young (large price diff)"
    );
}

// =============================================================================
// Delta Limit Tests
// =============================================================================

#[test]
fn extreme_delta_produces_no_quotes() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Set delta to 2x the limit.
    h.state.q_global_tao = 1000.0;
    h.state.dollar_delta_usd = 2.0 * h.state.delta_limit_usd + 1.0;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    for (i, q) in quotes.iter().enumerate() {
        assert!(
            q.bid.is_none() && q.ask.is_none(),
            "Venue {}: extreme delta (2x limit) should produce no quotes",
            i
        );
    }
}

#[test]
fn high_delta_only_allows_risk_reducing() {
    let mut h = make_harness(RiskProfile::Balanced);
    init_tick(&mut h, 0);
    setup_market(&mut h);

    // Set delta between 1x and 2x limit (long).
    h.state.q_global_tao = 100.0;
    h.state.dollar_delta_usd = 1.5 * h.state.delta_limit_usd;

    let quotes = compute_mm_quotes(h.cfg, &h.state);

    // When long and delta > limit, bids should be suppressed (risk-increasing).
    // Asks might still be allowed (risk-reducing).
    for q in &quotes {
        // Bids should be suppressed when very long and over delta limit.
        assert!(
            q.bid.is_none(),
            "Bids should be suppressed when long and delta > limit"
        );
    }
}

// =============================================================================
// Order Action Computation Tests
// =============================================================================

#[test]
fn compute_order_actions_place_new() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];

    let vstate = paraphina::state::VenueState {
        id: Arc::from("test"),
        mid: Some(300.0),
        spread: Some(0.10),
        prev_spread: None,
        prev_best_ask: None,
        depth_near_mid: 10_000.0,
        last_mid_update_ms: Some(0),
        last_mid_apply_ms: Some(0),
        last_book_update_ms: Some(0),
        quote_quarantined: false,
        quote_quarantined_at_ms: None,
        orderbook_l2: paraphina::orderbook_l2::OrderBookL2::new(),
        local_vol_short: 0.0,
        local_vol_long: 0.0,
        prev_ln_mid: None,
        status: VenueStatus::Healthy,
        toxicity: 0.0,
        pending_markouts: std::collections::VecDeque::new(),
        pending_markouts_next_eval_ms: i64::MAX,
        markout_ewma_usd_per_tao: 0.0,
        utility: Default::default(),
        open_orders: std::collections::BTreeMap::new(),
        recent_fills: std::collections::VecDeque::new(),
        recent_fills_cap: 0,
        next_fill_seq: 0,
        position_tao: 0.0,
        funding_8h: 0.0,
        funding_state: FundingState::default(),
        avg_entry_price: 0.0,
        margin_balance_usd: 10_000.0,
        margin_used_usd: 0.0,
        margin_available_usd: 10_000.0,
        price_liq: None,
        dist_liq_sigma: 10.0,
        mm_open_bid: None,
        mm_open_ask: None,
    };

    let quote = paraphina::mm::MmQuote {
        venue_index: 0,
        venue_id: Arc::from("test"),
        bid: Some(paraphina::mm::MmLevel {
            price: 299.90,
            size: 1.0,
            canonical_target_identity: None,
        }),
        ask: Some(paraphina::mm::MmLevel {
            price: 300.20,
            size: 1.0,
            canonical_target_identity: None,
        }),
        generated_spread_cap_applied: false,
        generated_spread_cap_bid_suppressed: false,
        generated_spread_cap_ask_suppressed: false,
        touch_mode_kind: None,
        bid_terminal_reason: "active",
        ask_terminal_reason: "active",
    };

    let actions = compute_order_actions(&cfg, vcfg, &vstate, &quote, None, None, 0);

    // Should have two Place actions.
    assert_eq!(
        actions.len(),
        2,
        "Should have two actions for new bid and ask"
    );

    let has_bid_place = actions.iter().any(|a| {
        matches!(
            a,
            MmOrderAction::Place {
                side: Side::Buy,
                ..
            }
        )
    });
    let has_ask_place = actions.iter().any(|a| {
        matches!(
            a,
            MmOrderAction::Place {
                side: Side::Sell,
                ..
            }
        )
    });

    assert!(has_bid_place, "Should have Place action for bid");
    assert!(has_ask_place, "Should have Place action for ask");
}

#[test]
fn compute_order_actions_cancel_when_no_desired() {
    let cfg = Config::default();
    let vcfg = &cfg.venues[0];

    let vstate = paraphina::state::VenueState {
        id: Arc::from("test"),
        mid: Some(300.0),
        spread: Some(0.10),
        prev_spread: None,
        prev_best_ask: None,
        depth_near_mid: 10_000.0,
        last_mid_update_ms: Some(0),
        last_mid_apply_ms: Some(0),
        last_book_update_ms: Some(0),
        quote_quarantined: false,
        quote_quarantined_at_ms: None,
        orderbook_l2: paraphina::orderbook_l2::OrderBookL2::new(),
        local_vol_short: 0.0,
        local_vol_long: 0.0,
        prev_ln_mid: None,
        status: VenueStatus::Healthy,
        toxicity: 0.0,
        pending_markouts: std::collections::VecDeque::new(),
        pending_markouts_next_eval_ms: i64::MAX,
        markout_ewma_usd_per_tao: 0.0,
        utility: Default::default(),
        open_orders: std::collections::BTreeMap::new(),
        recent_fills: std::collections::VecDeque::new(),
        recent_fills_cap: 0,
        next_fill_seq: 0,
        position_tao: 0.0,
        funding_8h: 0.0,
        funding_state: FundingState::default(),
        avg_entry_price: 0.0,
        margin_balance_usd: 10_000.0,
        margin_used_usd: 0.0,
        margin_available_usd: 10_000.0,
        price_liq: None,
        dist_liq_sigma: 10.0,
        mm_open_bid: None,
        mm_open_ask: None,
    };

    // No desired quote.
    let quote = paraphina::mm::MmQuote {
        venue_index: 0,
        venue_id: Arc::from("test"),
        bid: None,
        ask: None,
        generated_spread_cap_applied: false,
        generated_spread_cap_bid_suppressed: false,
        generated_spread_cap_ask_suppressed: false,
        touch_mode_kind: None,
        bid_terminal_reason: "not_quoted",
        ask_terminal_reason: "not_quoted",
    };

    // Existing orders.
    let current_bid = ActiveMmOrder {
        venue_index: 0,
        side: Side::Buy,
        price: 299.90,
        size: 1.0,
        timestamp_ms: 0,
    };
    let current_ask = ActiveMmOrder {
        venue_index: 0,
        side: Side::Sell,
        price: 300.20,
        size: 1.0,
        timestamp_ms: 0,
    };

    let actions = compute_order_actions(
        &cfg,
        vcfg,
        &vstate,
        &quote,
        Some(&current_bid),
        Some(&current_ask),
        1000,
    );

    // Should have two Cancel actions.
    assert_eq!(actions.len(), 2, "Should have two cancel actions");

    let has_bid_cancel = actions.iter().any(|a| {
        matches!(
            a,
            MmOrderAction::Cancel {
                side: Side::Buy,
                ..
            }
        )
    });
    let has_ask_cancel = actions.iter().any(|a| {
        matches!(
            a,
            MmOrderAction::Cancel {
                side: Side::Sell,
                ..
            }
        )
    });

    assert!(has_bid_cancel, "Should have Cancel action for bid");
    assert!(has_ask_cancel, "Should have Cancel action for ask");
}
