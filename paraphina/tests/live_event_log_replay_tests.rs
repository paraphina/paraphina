#![cfg(all(feature = "event_log", feature = "live", feature = "live_hyperliquid"))]

use std::fs::File;
use std::io::Read;
use std::sync::Mutex;

use paraphina::config::Config;
use paraphina::event_log::read_event_log;
use paraphina::live::connectors::hyperliquid::{
    HyperliquidAccountFixtureFeed, HyperliquidFixtureFeed,
};
use paraphina::live::runner::{
    replay_event_log, run_live_loop, LiveChannels, LiveRunMode, LiveTelemetry, LiveTelemetryStats,
};
use paraphina::live::shadow_adapter::ShadowAckAdapter;
use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink, TelemetrySinkHandle};
use sha2::{Digest, Sha256};
use tempfile::tempdir;
use tokio::sync::mpsc;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

fn hash_file(path: &std::path::Path) -> String {
    let mut file = File::open(path).expect("open file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read file");
    let mut hasher = Sha256::new();
    hasher.update(&buf);
    format!("{:x}", hasher.finalize())
}

#[tokio::test]
async fn live_event_log_replay_matches_telemetry() {
    let _guard = ENV_MUTEX.lock().unwrap();
    let dir = tempdir().expect("tempdir");
    let event_log = dir.path().join("event_log.jsonl");
    let telemetry_live = dir.path().join("telemetry_live.jsonl");
    let telemetry_replay = dir.path().join("telemetry_replay.jsonl");

    std::env::set_var("PARAPHINA_EVENT_LOG_PATH", &event_log);

    let cfg = Config::default();
    let (market_tx, market_rx) = mpsc::channel(512);
    let (account_tx, account_rx) = mpsc::channel(512);
    let (_exec_tx, exec_rx) = mpsc::channel(128);
    let (order_tx, mut order_rx) = mpsc::channel::<paraphina::live::runner::LiveOrderRequest>(256);

    let mut shadow = ShadowAckAdapter::new(&cfg);
    tokio::spawn(async move {
        while let Some(req) = order_rx.recv().await {
            let events =
                shadow.handle_intents(req.intents, req.action_batch.tick_index, req.now_ms);
            match req.response {
                paraphina::live::ResponseMode::Oneshot(tx) => { let _ = tx.send(events); }
                paraphina::live::ResponseMode::FireAndForget => {}
            }
        }
    });

    let fixture_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/hyperliquid");
    let feed = HyperliquidFixtureFeed::from_dir(&fixture_dir).expect("fixture feed");
    let account_feed =
        HyperliquidAccountFixtureFeed::from_dir(&fixture_dir).expect("account fixture feed");
    let market_tx_clone = market_tx.clone();
    tokio::spawn(async move {
        feed.run_ticks(market_tx_clone, 1, 1_000, 250, 8).await;
    });
    tokio::spawn(async move {
        account_feed.run_ticks(account_tx, 1_000, 250, 8).await;
    });

    let telemetry_cfg = TelemetryConfig {
        mode: TelemetryMode::Jsonl,
        path: Some(telemetry_live.clone()),
        append: false,
    };
    let telemetry = LiveTelemetry {
        sink: TelemetrySinkHandle::Sync(std::sync::Arc::new(std::sync::Mutex::new(
            TelemetrySink::from_config(telemetry_cfg),
        ))),
        shadow_mode: false,
        execution_mode: "replay",
        max_orders_per_tick: 200,
        stats: std::sync::Arc::new(LiveTelemetryStats::default()),
    };

    let channels = LiveChannels {
        market_rx,
        account_rx,
        exec_rx: Some(exec_rx),
        account_reconcile_tx: None,
        order_tx,
        order_snapshot_rx: None,
            shared_venue_ages: None,
    };

    let summary = run_live_loop(
        &cfg,
        channels,
        LiveRunMode::Step {
            start_ms: 1_000,
            step_ms: 250,
            ticks: 8,
        },
        Some(paraphina::live::runner::LiveRuntimeHooks {
            metrics: paraphina::live::ops::LiveMetrics::new(),
            health: paraphina::live::ops::HealthState::new(),
            telemetry: Some(telemetry),
        }),
    )
    .await;

    assert!(event_log.exists(), "event log should be written");
    assert!(!read_event_log(&event_log).unwrap().is_empty());

    let replay_summary = replay_event_log(&cfg, &event_log, &telemetry_replay, None);

    assert_eq!(summary.ticks_run, replay_summary.ticks_run);
    assert_eq!(summary.kill_switch, replay_summary.kill_switch);
    assert_eq!(
        summary.ready_market_count,
        replay_summary.ready_market_count
    );
    assert_eq!(
        summary.stale_market_count,
        replay_summary.stale_market_count
    );
    assert!((summary.local_vol_short_avg - replay_summary.local_vol_short_avg).abs() < 1e-9);
    assert!((summary.local_vol_long_avg - replay_summary.local_vol_long_avg).abs() < 1e-9);

    let hash_live = hash_file(&telemetry_live);
    let hash_replay = hash_file(&telemetry_replay);
    assert_eq!(hash_live, hash_replay, "telemetry hash must match");

    std::env::remove_var("PARAPHINA_EVENT_LOG_PATH");
}
