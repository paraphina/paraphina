#![cfg(feature = "live_hyperliquid")]

use std::path::PathBuf;

use paraphina::config::Config;
use paraphina::live::connectors::hyperliquid::HyperliquidFixtureFeed;
use paraphina::live::ops::{HealthState, LiveMetrics};
use paraphina::live::runner::{
    run_live_loop, LiveChannels, LiveOrderRequest, LiveRunMode, LiveRuntimeHooks,
};
use paraphina::live::shadow_adapter::ShadowAckAdapter;
use paraphina::live::types::MarketDataEvent;
use paraphina::state::GlobalState;
use tokio::sync::mpsc;

fn market_event_timestamp(event: &MarketDataEvent) -> i64 {
    match event {
        MarketDataEvent::L2Snapshot(snapshot) => snapshot.timestamp_ms,
        MarketDataEvent::L2Delta(delta) => delta.timestamp_ms,
        MarketDataEvent::Trade(trade) => trade.timestamp_ms,
        MarketDataEvent::FundingUpdate(update) => update.timestamp_ms,
    }
}

fn compute_expected_local_vols(cfg: &Config, events: &[MarketDataEvent]) -> (f64, f64, usize) {
    let mut state = GlobalState::new(cfg);
    let venue = state.venues.get_mut(0).expect("venue");
    let alpha_short = cfg.volatility.fv_vol_alpha_short;
    let alpha_long = cfg.volatility.fv_vol_alpha_long;
    let max_levels = cfg.book.depth_levels.max(1) as usize;
    let mut change_count = 0usize;
    let mut prev_mid: Option<f64> = None;
    for event in events {
        match event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                let _ = venue.apply_l2_snapshot(
                    &snapshot.bids,
                    &snapshot.asks,
                    snapshot.seq,
                    snapshot.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                );
            }
            MarketDataEvent::L2Delta(delta) => {
                let _ = venue.apply_l2_delta(
                    &delta.changes,
                    delta.seq,
                    delta.timestamp_ms,
                    max_levels,
                    alpha_short,
                    alpha_long,
                );
            }
            MarketDataEvent::Trade(_) | MarketDataEvent::FundingUpdate(_) => {}
        }
        if let Some(mid) = venue.mid {
            if let Some(prev) = prev_mid {
                if (mid - prev).abs() > 1e-12 {
                    change_count += 1;
                }
            }
            prev_mid = Some(mid);
        }
    }

    (venue.local_vol_short, venue.local_vol_long, change_count)
}

#[tokio::test]
async fn live_runner_consumes_hyperliquid_fixtures() {
    // Disable L2 coalescing so every event is applied individually,
    // matching compute_expected_local_vols' sequential application.
    std::env::set_var("PARAPHINA_L2_DELTA_COALESCE", "0");
    std::env::set_var("PARAPHINA_L2_SNAPSHOT_COALESCE", "0");

    let mut cfg = Config::default();
    cfg.venues = vec![cfg.venues[0].clone()];
    cfg.book.min_healthy_for_kf = 1;
    cfg.main_loop_interval_ms = 250;
    cfg.hedge_loop_interval_ms = 250;
    cfg.risk_loop_interval_ms = 250;

    let (market_tx, market_rx) = mpsc::channel(1024);
    let (_account_tx, account_rx) = mpsc::channel(128);
    let (_exec_tx, exec_rx) = mpsc::channel(128);
    let (_order_snapshot_tx, order_snapshot_rx) = mpsc::channel(128);
    let (order_tx, mut order_rx) = mpsc::channel::<LiveOrderRequest>(256);

    let fixture_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/hyperliquid");
    let feed = HyperliquidFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
    let expected_feed = HyperliquidFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
    let start_ms = 1_000;
    let step_ms = 250;
    let ticks = 20;
    let (expected_tx, mut expected_rx) = mpsc::channel(1024);
    expected_feed
        .run_ticks(expected_tx, 0, start_ms, step_ms, ticks)
        .await;
    let mut expected_events = Vec::new();
    while let Some(event) = expected_rx.recv().await {
        expected_events.push(event);
    }
    let feed_task = tokio::spawn(async move {
        feed.run_ticks(market_tx, 0, start_ms, step_ms, ticks).await;
    });
    tokio::task::yield_now().await;

    let cfg_clone = cfg.clone();
    tokio::spawn(async move {
        let mut shadow = ShadowAckAdapter::new(&cfg_clone);
        while let Some(req) = order_rx.recv().await {
            let LiveOrderRequest {
                intents,
                action_batch,
                now_ms,
                response,
            } = req;
            let events = shadow.handle_intents(intents, action_batch.tick_index, now_ms);
            match response {
                paraphina::live::ResponseMode::Oneshot(tx) => { let _ = tx.send(events); }
                paraphina::live::ResponseMode::FireAndForget => {}
            }
        }
    });

    let hooks = LiveRuntimeHooks {
        metrics: LiveMetrics::new(),
        health: HealthState::new(),
        telemetry: None,
    };
    let summary = run_live_loop(
        &cfg,
        LiveChannels {
            market_rx,
            account_rx,
            exec_rx: Some(exec_rx),
            account_reconcile_tx: None,
            order_tx,
            order_snapshot_rx: Some(order_snapshot_rx),
        },
        LiveRunMode::Step {
            start_ms,
            step_ms,
            ticks,
        },
        Some(hooks),
    )
    .await;
    let _ = feed_task.await;

    assert!(summary.fv_available);
    assert_eq!(summary.ready_market_count, 1);
    assert_eq!(summary.stale_market_count, 0);
    let (expected_short, expected_long, change_count) =
        compute_expected_local_vols(&cfg, &expected_events);
    let tol = 1e-9;
    assert!(
        (summary.local_vol_short_avg - expected_short).abs() < tol,
        "short vol mismatch: expected {expected_short} got {}",
        summary.local_vol_short_avg
    );
    assert!(
        (summary.local_vol_long_avg - expected_long).abs() < tol,
        "long vol mismatch: expected {expected_long} got {}",
        summary.local_vol_long_avg
    );
    assert!(
        change_count > 0,
        "fixture feed must produce at least one mid change"
    );
    assert!(summary.local_vol_short_avg > 0.0);
    assert!(summary.local_vol_long_avg > 0.0);
}
