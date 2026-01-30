#![cfg(feature = "live_hyperliquid")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use paraphina::config::Config;
use paraphina::live::connectors::hyperliquid::HyperliquidFixtureFeed;
use paraphina::live::ops::{HealthState, LiveMetrics};
use paraphina::live::runner::{
    run_live_loop, LiveChannels, LiveOrderRequest, LiveRunMode, LiveRuntimeHooks,
};
use paraphina::live::shadow_adapter::ShadowAckAdapter;
use paraphina::live::{LiveTelemetry, LiveTelemetryStats};
use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink};
use tempfile::tempdir;
use tokio::sync::mpsc;

#[tokio::test]
async fn live_telemetry_contract_passes_fixture_run() {
    std::env::set_var("PARAPHINA_MARKET_RX_STATS", "1");

    let mut cfg = Config::default();
    cfg.venues = vec![cfg.venues[0].clone()];
    cfg.book.min_healthy_for_kf = 1;
    cfg.main_loop_interval_ms = 250;
    cfg.hedge_loop_interval_ms = 250;
    cfg.risk_loop_interval_ms = 250;

    let temp = tempdir().expect("tempdir");
    let telemetry_path = temp.path().join("telemetry.jsonl");
    let telemetry = LiveTelemetry {
        sink: Arc::new(Mutex::new(TelemetrySink::from_config(TelemetryConfig {
            mode: TelemetryMode::Jsonl,
            path: Some(telemetry_path.clone()),
            append: false,
        }))),
        shadow_mode: true,
        execution_mode: "shadow",
        max_orders_per_tick: 50,
        stats: Arc::new(Mutex::new(LiveTelemetryStats::default())),
    };

    let (market_tx, market_rx) = mpsc::channel(1024);
    let (_account_tx, account_rx) = mpsc::channel(128);
    let (_exec_tx, exec_rx) = mpsc::channel(128);
    let (_order_snapshot_tx, order_snapshot_rx) = mpsc::channel(128);
    let (order_tx, mut order_rx) = mpsc::channel::<LiveOrderRequest>(256);

    let fixture_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/hyperliquid");
    let feed = HyperliquidFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
    let start_ms = 1_000;
    let step_ms = 250;
    let ticks = 10;
    feed.run_ticks(market_tx, 0, start_ms, step_ms, ticks).await;

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
            let _ = response.send(events);
        }
    });

    let hooks = LiveRuntimeHooks {
        metrics: LiveMetrics::new(),
        health: HealthState::new(),
        telemetry: Some(telemetry.clone()),
    };

    let _ = run_live_loop(
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

    if let Ok(mut guard) = telemetry.sink.lock() {
        guard.flush();
    }

    let text = std::fs::read_to_string(&telemetry_path).expect("read telemetry");
    assert!(
        text.contains("\"market_rx_stats\""),
        "expected market_rx_stats in telemetry"
    );

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let status = Command::new("python3")
        .current_dir(&repo_root)
        .arg("tools/check_telemetry_contract.py")
        .arg(&telemetry_path)
        .status()
        .expect("telemetry contract command");
    assert!(status.success());

    std::env::remove_var("PARAPHINA_MARKET_RX_STATS");
}
