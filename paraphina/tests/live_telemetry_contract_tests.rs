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
use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink, TelemetrySinkHandle};
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
        sink: TelemetrySinkHandle::Sync(Arc::new(Mutex::new(TelemetrySink::from_config(
            TelemetryConfig {
                mode: TelemetryMode::Jsonl,
                path: Some(telemetry_path.clone()),
                append: false,
            },
        )))),
        shadow_mode: true,
        execution_mode: "shadow",
        max_orders_per_tick: 50,
        stats: Arc::new(LiveTelemetryStats::default()),
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
                transport_hint: _,
                response,
            } = req;
            let events = shadow.handle_intents(intents, action_batch.tick_index, now_ms);
            match response {
                paraphina::live::ResponseMode::Oneshot(tx) => {
                    let _ = tx.send(events);
                }
                paraphina::live::ResponseMode::FireAndForget => {}
            }
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
            priority_order_tx: order_tx.clone(),
            order_tx,
            order_snapshot_rx: Some(order_snapshot_rx),
            shared_venue_ages: None,
        },
        LiveRunMode::Step {
            start_ms,
            step_ms,
            ticks,
        },
        Some(hooks),
    )
    .await;

    if let TelemetrySinkHandle::Sync(ref sink) = telemetry.sink {
        if let Ok(mut guard) = sink.lock() {
            guard.flush();
        }
    }

    let text = std::fs::read_to_string(&telemetry_path).expect("read telemetry");
    assert!(
        text.contains("\"market_rx_stats\""),
        "expected market_rx_stats in telemetry"
    );
    let has_mm_order_management = text.lines().any(|line| {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return false;
        };
        value
            .get("mm_order_management")
            .and_then(|v| v.as_object())
            .is_some_and(|summary| {
                summary.contains_key("keep_count")
                    && summary.contains_key("replace_count")
                    && summary.contains_key("decision_records")
                    && summary.contains_key("aster_touch_offside_fastpath_count")
                    && summary.contains_key("aster_touch_offside_nearmiss_by_reason")
                    && summary.contains_key("touch_risk_hysteresis_count")
                    && summary.contains_key("touch_risk_size_band_count")
                    && summary.contains_key("touch_risk_nearmiss_by_reason")
                    && summary.contains_key("compression_edge_hysteresis_count")
                    && summary.contains_key("compression_edge_nearmiss_by_reason")
                    && summary.contains_key("keep_by_venue_role")
                    && summary.contains_key("replace_by_venue_role")
            })
    });
    assert!(
        has_mm_order_management,
        "expected mm_order_management summary in telemetry"
    );
    let has_decision_ids = text.lines().any(|line| {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return false;
        };
        value
            .get("orders")
            .and_then(|v| v.as_array())
            .is_some_and(|orders| {
                orders.iter().any(|order| {
                    order
                        .get("decision_id")
                        .and_then(|v| v.as_str())
                        .is_some_and(|decision_id| !decision_id.is_empty())
                })
            })
    });
    assert!(
        has_decision_ids,
        "expected decision_id lineage on telemetry order records"
    );
    let has_venue_health_fields = text.lines().any(|line| {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return false;
        };
        value
            .get("venue_toxicity_bootstrap_pending")
            .and_then(|v| v.as_array())
            .is_some()
            && value
                .get("venue_generated_quote_spread_bps")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_generated_quote_spread_cap_applied")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_bid_distance_to_touch_bps")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_ask_distance_to_touch_bps")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_touch_mode_applied")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_touch_offset_ticks")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_health_api_errors")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_health_stale_count")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_health_dev_breaches")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_health_disable_reason")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_health_last_error_source")
                .and_then(|v| v.as_array())
                .is_some()
            && value
                .get("venue_health_last_error_message")
                .and_then(|v| v.as_array())
                .is_some()
    });
    assert!(
        has_venue_health_fields,
        "expected venue health disable attribution fields in telemetry"
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
