//! Live trading skeleton binary (feature-gated).
//!
//! This binary wires the live cache, event model, and strategy loop together
//! without any external network connectors.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::{Parser, ValueEnum};
use paraphina::config::{resolve_effective_profile, Config};
use paraphina::live::ops::{
    default_audit_dir, format_startup_log, start_metrics_server, EnvSecretProvider,
    HealthState, LiveMetrics, SecretProvider, write_audit_files,
};
use paraphina::live::instrument::{InstrumentSpec, validate_specs};
use paraphina::live::orderbook_l2::BookLevel;
use paraphina::live::runner::{run_live_loop, LiveChannels, LiveOrderRequest, LiveRunMode, LiveRuntimeHooks};
use paraphina::io::GatewayPolicy;
use paraphina::live::gateway::{LiveGateway, LiveRestClient};
use paraphina::live::shadow_adapter::ShadowAckAdapter;
use paraphina::live::{resolve_effective_trade_mode, LiveTelemetry, LiveTelemetryStats, TradeMode};
use paraphina::live::types::L2Snapshot;
use paraphina::live::venues::{canonical_venue_ids, roadmap_b_enabled};
use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink};

use tokio::sync::mpsc;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TradeModeArg {
    Shadow,
    Paper,
    Testnet,
    Live,
}

impl From<TradeModeArg> for TradeMode {
    fn from(value: TradeModeArg) -> Self {
        match value {
            TradeModeArg::Shadow => TradeMode::Shadow,
            TradeModeArg::Paper => TradeMode::Paper,
            TradeModeArg::Testnet => TradeMode::Testnet,
            TradeModeArg::Live => TradeMode::Live,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ConnectorArg {
    Mock,
    Hyperliquid,
    HyperliquidFixture,
    Lighter,
    Extended,
    Aster,
    Paradex,
}

impl ConnectorArg {
    fn as_str(&self) -> &'static str {
        match self {
            ConnectorArg::Mock => "mock",
            ConnectorArg::Hyperliquid => "hyperliquid",
            ConnectorArg::HyperliquidFixture => "hyperliquid_fixture",
            ConnectorArg::Lighter => "lighter",
            ConnectorArg::Extended => "extended",
            ConnectorArg::Aster => "aster",
            ConnectorArg::Paradex => "paradex",
        }
    }

    fn parse_env(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mock" => Some(ConnectorArg::Mock),
            "hyperliquid" | "hl" => Some(ConnectorArg::Hyperliquid),
            "hyperliquid_fixture" | "hl_fixture" | "fixture" => Some(ConnectorArg::HyperliquidFixture),
            "lighter" => Some(ConnectorArg::Lighter),
            "extended" => Some(ConnectorArg::Extended),
            "aster" => Some(ConnectorArg::Aster),
            "paradex" => Some(ConnectorArg::Paradex),
            _ => None,
        }
    }

    fn all() -> &'static [ConnectorArg] {
        &[
            ConnectorArg::Mock,
            ConnectorArg::Hyperliquid,
            ConnectorArg::HyperliquidFixture,
            ConnectorArg::Lighter,
            ConnectorArg::Extended,
            ConnectorArg::Aster,
            ConnectorArg::Paradex,
        ]
    }

    fn roadmap_b_venue_id(&self) -> Option<&'static str> {
        match self {
            ConnectorArg::Hyperliquid | ConnectorArg::HyperliquidFixture => Some("hyperliquid"),
            ConnectorArg::Lighter => Some("lighter"),
            ConnectorArg::Extended => Some("extended"),
            ConnectorArg::Aster => Some("aster"),
            ConnectorArg::Paradex => Some("paradex"),
            ConnectorArg::Mock => None,
        }
    }

    fn roadmap_b_selectable_venues() -> Vec<&'static str> {
        let mut available = std::collections::BTreeSet::new();
        for connector in Self::all() {
            if let Some(venue_id) = connector.roadmap_b_venue_id() {
                available.insert(venue_id);
            }
        }
        canonical_venue_ids()
            .iter()
            .filter(|venue_id| available.contains(*venue_id))
            .copied()
            .collect()
    }
}

#[derive(Debug, Parser)]
#[command(name = "paraphina_live", about = "Paraphina live runner (shadow-safe by default)", version)]
struct Args {
    /// Trade mode: shadow (default), paper, testnet, live.
    #[arg(long, value_enum)]
    trade_mode: Option<TradeModeArg>,
    /// Connector to use: mock (default), hyperliquid, hyperliquid_fixture, lighter, extended, aster, paradex.
    #[arg(long, value_enum)]
    connector: Option<ConnectorArg>,
    /// Output directory for telemetry/audit artifacts.
    #[arg(long)]
    out_dir: Option<String>,
}

fn resolve_connector(cli: Option<ConnectorArg>) -> ConnectorArg {
    if let Some(connector) = cli {
        return connector;
    }
    if let Ok(env_val) = std::env::var("PARAPHINA_LIVE_CONNECTOR") {
        if let Some(connector) = ConnectorArg::parse_env(&env_val) {
            return connector;
        }
        if !env_val.is_empty() {
            eprintln!(
                "paraphina_live | warn=invalid_connector value={:?} fallback=mock",
                env_val
            );
        }
    }
    ConnectorArg::Mock
}

fn resolve_out_dir(cli: Option<String>) -> Option<std::path::PathBuf> {
    if let Some(val) = cli {
        if !val.trim().is_empty() {
            return Some(std::path::PathBuf::from(val));
        }
    }
    if let Ok(val) = std::env::var("PARAPHINA_LIVE_OUT_DIR") {
        if !val.trim().is_empty() {
            return Some(std::path::PathBuf::from(val));
        }
    }
    None
}

fn enforce_roadmap_b_gate() {
    if !roadmap_b_enabled() {
        return;
    }
    let selectable = ConnectorArg::roadmap_b_selectable_venues();
    let required = canonical_venue_ids();
    if selectable.len() < required.len() {
        let missing: Vec<&str> = required
            .iter()
            .filter(|venue_id| !selectable.contains(venue_id))
            .copied()
            .collect();
        eprintln!(
            "paraphina_live | error=roadmap_b_gate_failed missing={:?} selectable={:?}",
            missing, selectable
        );
        std::process::exit(2);
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    enforce_roadmap_b_gate();
    let effective = resolve_effective_profile(None, None);
    effective.log_startup();
    let cfg = Config::from_env_or_profile(effective.profile);
    let build_info = paraphina::BuildInfo::capture();
    let trade_mode = resolve_effective_trade_mode(args.trade_mode.map(TradeMode::from));
    trade_mode.log_startup();
    let connector = resolve_connector(args.connector);
    let out_dir = resolve_out_dir(args.out_dir);
    let metrics_addr = std::env::var("PARAPHINA_LIVE_METRICS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9898".to_string());
    let startup_log = format_startup_log(
        &cfg,
        &build_info,
        trade_mode.trade_mode,
        connector.as_str(),
        &metrics_addr,
    );
    println!("{startup_log}");
    eprintln!(
        "paraphina_live | trade_mode={} connector={}",
        trade_mode.trade_mode.as_str(),
        connector.as_str()
    );

    let audit_dir = out_dir.clone().unwrap_or_else(default_audit_dir);
    if let Err(err) = std::fs::create_dir_all(&audit_dir) {
        eprintln!("paraphina_live | audit_dir_create_error={err}");
    }
    if let Err(err) = write_audit_files(&audit_dir, &cfg, &build_info) {
        eprintln!("paraphina_live | audit_write_error={err}");
    }
    let specs = InstrumentSpec::from_config(&cfg);
    if let Err(errors) = validate_specs(&specs) {
        for err in errors {
            eprintln!("paraphina_live | instrument_spec_error={err}");
        }
    }

    let metrics = LiveMetrics::new();
    let health = HealthState::new();
    start_metrics_server(&metrics_addr, metrics.clone(), health.clone(), audit_dir.clone());

    let secrets = EnvSecretProvider::default();
    if secrets.get("PARAPHINA_LIVE_MODE").is_some() {
        // Secret provider is wired for future use.
    }

    let (market_tx, market_rx) = mpsc::channel::<paraphina::live::types::MarketDataEvent>(1024);
    let (_account_tx, account_rx) = mpsc::channel::<paraphina::live::types::AccountEvent>(256);
    let (_exec_tx, exec_rx) = mpsc::channel::<paraphina::live::types::ExecutionEvent>(512);
    let (_order_snapshot_tx, order_snapshot_rx) =
        mpsc::channel::<paraphina::live::types::OrderSnapshot>(128);

    if let Ok(val) = std::env::var("PARAPHINA_LIVE_RECONCILE_MS") {
        if let Ok(ms) = val.parse::<u64>() {
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(ms.max(100)));
                loop {
                    interval.tick().await;
                    // Connector-provided snapshots should be sent on this channel.
                    // Stub binary logs only.
                    println!("paraphina_live | reconcile_tick_ms={}", ms);
                }
            });
        }
    }
    let (order_tx, mut order_rx) = mpsc::channel::<LiveOrderRequest>(256);

    if matches!(connector, ConnectorArg::Mock) {
        let market_tx_clone = market_tx.clone();
        tokio::spawn(async move {
            let mut seq: u64 = 0;
            let mut mid = 100.0;
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            loop {
                interval.tick().await;
                seq += 1;
                mid += if seq % 2 == 0 { 0.1 } else { -0.1 };
                let bids = vec![
                    BookLevel { price: mid - 0.5, size: 5.0 },
                    BookLevel { price: mid - 1.0, size: 5.0 },
                ];
                let asks = vec![
                    BookLevel { price: mid + 0.5, size: 5.0 },
                    BookLevel { price: mid + 1.0, size: 5.0 },
                ];
                let snapshot = L2Snapshot {
                    venue_index: 0,
                    venue_id: "mock".to_string(),
                    seq,
                    timestamp_ms: 0,
                    bids,
                    asks,
                };
                let _ = market_tx_clone
                    .send(paraphina::live::types::MarketDataEvent::L2Snapshot(snapshot))
                    .await;
            }
        });
    }

    let exec_enabled = std::env::var("PARAPHINA_LIVE_EXEC_ENABLE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mut exec_client: Option<std::sync::Arc<dyn LiveRestClient>> = None;
    let mut exec_supported = false;

    match connector {
        ConnectorArg::Hyperliquid => {
            #[cfg(feature = "live_hyperliquid")]
            {
                let hl_cfg = paraphina::live::connectors::hyperliquid::HyperliquidConfig::from_env();
                let mut hl = paraphina::live::connectors::hyperliquid::HyperliquidConnector::new(
                    hl_cfg.clone(),
                    market_tx.clone(),
                    _exec_tx.clone(),
                );
                if trade_mode.trade_mode != TradeMode::Shadow {
                    if hl_cfg.vault_address.is_some() {
                        let account_tx = _account_tx.clone();
                        hl = hl.with_account_tx(account_tx);
                        // account_tx wired below
                    } else {
                        eprintln!("paraphina_live | account_snapshots_disabled=true reason=missing_hl_vault_address");
                        send_unavailable_account_snapshot(&_account_tx, &cfg);
                    }
                } else {
                    eprintln!("paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow");
                    send_unavailable_account_snapshot(&_account_tx, &cfg);
                }
                let hl_arc = Arc::new(hl);
                exec_supported = true;
                if exec_enabled && trade_mode.trade_mode != TradeMode::Shadow {
                    if hl_cfg.private_key_hex.is_some() {
                        exec_client = Some(hl_arc.clone());
                    } else {
                        eprintln!("paraphina_live | exec_disabled=true reason=missing_hl_private_key");
                    }
                }
                if trade_mode.trade_mode != TradeMode::Shadow && hl_cfg.vault_address.is_some() {
                    let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(5_000);
                    let hl_poll = hl_arc.clone();
                    tokio::spawn(async move {
                        hl_poll.run_account_polling(poll_ms).await;
                    });
                    if hl_cfg.private_key_hex.is_some() {
                        let hl_private = hl_arc.clone();
                        tokio::spawn(async move {
                            hl_private.run_private_ws().await;
                        });
                    } else {
                        eprintln!("paraphina_live | private_ws_disabled=true reason=missing_hl_private_key");
                    }
                }
                let hl_public = hl_arc.clone();
                tokio::spawn(async move {
                    hl_public.run_public_ws().await;
                });
            }
            #[cfg(not(feature = "live_hyperliquid"))]
            {
                eprintln!("paraphina_live | error=connector_unavailable connector=hyperliquid");
                return;
            }
        }
        ConnectorArg::HyperliquidFixture => {
            #[cfg(feature = "live_hyperliquid")]
            {
                let fixture_dir = std::env::var("HL_FIXTURE_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| std::path::PathBuf::from("./tests/fixtures/hyperliquid"));
                match paraphina::live::connectors::hyperliquid::HyperliquidFixtureFeed::from_dir(
                    &fixture_dir,
                ) {
                    Ok(feed) => {
                        tokio::spawn(async move {
                            feed.run_ticks(market_tx.clone(), 1_000, 250, 200).await;
                        });
                    }
                    Err(err) => {
                        eprintln!(
                            "paraphina_live | error=fixture_dir_unreadable dir={} err={}",
                            fixture_dir.display(),
                            err
                        );
                    }
                }
                if let Ok(feed) = paraphina::live::connectors::hyperliquid::HyperliquidAccountFixtureFeed::from_dir(&fixture_dir) {
                    let account_tx = _account_tx.clone();
                    tokio::spawn(async move {
                        feed.run_ticks(account_tx, 1_000, 250, 200).await;
                    });
                }
                exec_supported = true;
            }
            #[cfg(not(feature = "live_hyperliquid"))]
            {
                eprintln!("paraphina_live | error=connector_unavailable connector=hyperliquid_fixture");
                return;
            }
        }
        ConnectorArg::Lighter => {
            #[cfg(feature = "live_lighter")]
            {
                let lighter_cfg = paraphina::live::connectors::lighter::LighterConfig::from_env();
                let mut lighter = paraphina::live::connectors::lighter::LighterConnector::new(
                    lighter_cfg.clone(),
                    market_tx.clone(),
                    _exec_tx.clone(),
                );
                if trade_mode.trade_mode != TradeMode::Shadow {
                    if lighter_cfg.paper_mode {
                        eprintln!("paraphina_live | account_snapshots_disabled=true reason=lighter_paper_mode");
                        send_unavailable_account_snapshot(&_account_tx, &cfg);
                    } else {
                        lighter = lighter.with_account_tx(_account_tx.clone());
                        // account_tx wired below
                    }
                } else {
                    eprintln!("paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow");
                    send_unavailable_account_snapshot(&_account_tx, &cfg);
                }
                let lighter_arc = Arc::new(lighter);
                exec_supported = true;
                if exec_enabled && trade_mode.trade_mode != TradeMode::Shadow {
                    if lighter_cfg.paper_mode {
                        eprintln!("paraphina_live | exec_disabled=true reason=lighter_paper_mode");
                    } else {
                        exec_client = Some(lighter_arc.clone());
                    }
                }
                if trade_mode.trade_mode != TradeMode::Shadow && !lighter_cfg.paper_mode {
                    let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(5_000);
                    let lighter_poll = lighter_arc.clone();
                    tokio::spawn(async move {
                        lighter_poll.run_account_polling(poll_ms).await;
                    });
                }
                let lighter_public = lighter_arc.clone();
                tokio::spawn(async move {
                    lighter_public.run_public_ws().await;
                });
            }
            #[cfg(not(feature = "live_lighter"))]
            {
                eprintln!("paraphina_live | error=connector_unavailable connector=lighter");
                return;
            }
        }
        ConnectorArg::Extended | ConnectorArg::Aster | ConnectorArg::Paradex => {
            eprintln!(
                "paraphina_live | error=connector_unavailable connector={}",
                connector.as_str()
            );
            return;
        }
        ConnectorArg::Mock => {}
    }

    let exec_enabled = exec_enabled
        && trade_mode.trade_mode != TradeMode::Shadow
        && exec_supported
        && exec_client.is_some();
    if trade_mode.trade_mode != TradeMode::Shadow && !exec_enabled {
        eprintln!(
            "paraphina_live | trade_mode={} | exec_disabled=true | falling_back=shadow (set PARAPHINA_LIVE_EXEC_ENABLE=1 and provide keys)",
            trade_mode.trade_mode.as_str()
        );
    }

    let exec_trade_mode = trade_mode.trade_mode;
    let exec_cfg = cfg.clone();
    let exec_enabled_flag = exec_enabled;
    let exec_client = exec_client.clone();
    let exec_metrics = metrics.clone();
    tokio::spawn(async move {
        let mut shadow = ShadowAckAdapter::new(&exec_cfg);
        let mut live_gateway = if exec_enabled_flag {
            match LiveGateway::new(
                &exec_cfg,
                exec_client.expect("exec client"),
                GatewayPolicy::from_env(),
                Some(exec_metrics.clone()),
                exec_trade_mode,
            ) {
                Ok(gw) => Some(gw),
                Err(err) => {
                    eprintln!("paraphina_live | exec_gateway_error={} fallback=shadow", err.message);
                    None
                }
            }
        } else {
            None
        };
        let mut exec_seq: u64 = 0;
        while let Some(req) = order_rx.recv().await {
            let LiveOrderRequest {
                intents,
                action_batch,
                now_ms,
                response,
            } = req;
            let events = if let Some(gateway) = live_gateway.as_mut() {
                handle_live_gateway_intents(
                    gateway,
                    intents,
                    action_batch.tick_index,
                    now_ms,
                    &mut exec_seq,
                )
                .await
            } else {
                shadow.handle_intents(intents, action_batch.tick_index, now_ms)
            };
            let _ = response.send(events);
        }
    });

    let channels = LiveChannels {
        market_rx,
        account_rx,
        exec_rx: Some(exec_rx),
        account_reconcile_tx: None,
        order_tx,
        order_snapshot_rx: Some(order_snapshot_rx),
    };
    let max_orders_per_tick = std::env::var("PARAPHINA_LIVE_TELEMETRY_MAX_ORDERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(200);
    let mut telemetry_path = std::env::var("PARAPHINA_TELEMETRY_PATH")
        .ok()
        .map(std::path::PathBuf::from);
    if telemetry_path.is_none() {
        if let Some(out_dir) = out_dir.as_ref() {
            telemetry_path = Some(out_dir.join("telemetry.jsonl"));
        }
    }
    let telemetry_cfg = TelemetryConfig {
        mode: TelemetryMode::from_env(),
        path: telemetry_path,
        append: TelemetryConfig::append_from_env(),
    };
    let telemetry = LiveTelemetry {
        sink: Arc::new(Mutex::new(TelemetrySink::from_config(telemetry_cfg))),
        shadow_mode: trade_mode.trade_mode == TradeMode::Shadow,
        max_orders_per_tick,
        stats: Arc::new(Mutex::new(LiveTelemetryStats::default())),
    };
    let hooks = LiveRuntimeHooks {
        metrics,
        health,
        telemetry: Some(telemetry.clone()),
    };
    let max_ticks = std::env::var("PARAPHINA_LIVE_MAX_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let summary = run_live_loop(
        &cfg,
        channels,
        LiveRunMode::Realtime {
            interval_ms: cfg.main_loop_interval_ms as u64,
            max_ticks,
        },
        Some(hooks),
    )
    .await;

    if let Some(out_dir) = out_dir {
        write_summary(
            &out_dir,
            &cfg,
            trade_mode.trade_mode,
            connector.as_str(),
            &summary,
            telemetry.stats.clone(),
        );
    }
}

async fn handle_live_gateway_intents<C: LiveRestClient>(
    gateway: &mut LiveGateway<C>,
    intents: Vec<paraphina::types::OrderIntent>,
    tick: u64,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    let mut events = Vec::new();
    for intent in intents {
        let mut out = match intent {
            paraphina::types::OrderIntent::Place(place) => {
                handle_live_gateway_place(gateway, place, tick, now_ms, seq).await
            }
            paraphina::types::OrderIntent::Cancel(cancel) => {
                handle_live_gateway_cancel(gateway, cancel, tick, now_ms, seq).await
            }
            paraphina::types::OrderIntent::CancelAll(cancel_all) => {
                handle_live_gateway_cancel_all(gateway, cancel_all, tick, now_ms, seq).await
            }
            paraphina::types::OrderIntent::Replace(replace) => {
                let mut out = handle_live_gateway_cancel(
                    gateway,
                    paraphina::types::CancelOrderIntent {
                        venue_index: replace.venue_index,
                        venue_id: replace.venue_id.clone(),
                        order_id: replace.order_id.clone(),
                    },
                    tick,
                    now_ms,
                    seq,
                )
                .await;
                let mut out2 = handle_live_gateway_place(
                    gateway,
                    paraphina::types::PlaceOrderIntent {
                        venue_index: replace.venue_index,
                        venue_id: replace.venue_id.clone(),
                        side: replace.side,
                        price: replace.price,
                        size: replace.size,
                        purpose: replace.purpose,
                        time_in_force: replace.time_in_force,
                        post_only: replace.post_only,
                        reduce_only: replace.reduce_only,
                        client_order_id: replace.client_order_id.clone(),
                    },
                    tick,
                    now_ms,
                    seq,
                )
                .await;
                out.append(&mut out2);
                out
            }
        };
        events.append(&mut out);
    }
    events
}

async fn handle_live_gateway_place<C: LiveRestClient>(
    gateway: &mut LiveGateway<C>,
    place: paraphina::types::PlaceOrderIntent,
    tick: u64,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    use paraphina::live::types::{ExecutionEvent, OrderAccepted, OrderRejected};
    let mut events = Vec::new();
    let res = gateway
        .submit_intent(&paraphina::types::OrderIntent::Place(place.clone()), tick, now_ms)
        .await;
    match res {
        Ok(resp) => {
            *seq = seq.wrapping_add(1);
            events.push(ExecutionEvent::OrderAccepted(OrderAccepted {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: *seq,
                timestamp_ms: now_ms,
                order_id: resp.order_id.clone().unwrap_or_else(|| {
                    place
                        .client_order_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string())
                }),
                client_order_id: place.client_order_id.clone(),
                side: place.side,
                price: place.price,
                size: place.size,
                purpose: place.purpose,
            }));
        }
        Err(err) => {
            *seq = seq.wrapping_add(1);
            events.push(ExecutionEvent::OrderRejected(OrderRejected {
                venue_index: place.venue_index,
                venue_id: place.venue_id.to_string(),
                seq: *seq,
                timestamp_ms: now_ms,
                order_id: place.client_order_id.clone(),
                reason: err.message.clone(),
            }));
        }
    }
    events
}

async fn handle_live_gateway_cancel<C: LiveRestClient>(
    gateway: &mut LiveGateway<C>,
    cancel: paraphina::types::CancelOrderIntent,
    tick: u64,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    use paraphina::live::types::{CancelAccepted, CancelRejected, ExecutionEvent};
    let mut events = Vec::new();
    let res =
        gateway.submit_intent(&paraphina::types::OrderIntent::Cancel(cancel.clone()), tick, now_ms).await;
    match res {
        Ok(_resp) => {
            *seq = seq.wrapping_add(1);
            events.push(ExecutionEvent::CancelAccepted(CancelAccepted {
                venue_index: cancel.venue_index,
                venue_id: cancel.venue_id.to_string(),
                seq: *seq,
                timestamp_ms: now_ms,
                order_id: cancel.order_id.clone(),
            }));
        }
        Err(err) => {
            *seq = seq.wrapping_add(1);
            events.push(ExecutionEvent::CancelRejected(CancelRejected {
                venue_index: cancel.venue_index,
                venue_id: cancel.venue_id.to_string(),
                seq: *seq,
                timestamp_ms: now_ms,
                order_id: Some(cancel.order_id.clone()),
                reason: err.message.clone(),
            }));
        }
    }
    events
}

async fn handle_live_gateway_cancel_all<C: LiveRestClient>(
    gateway: &mut LiveGateway<C>,
    cancel_all: paraphina::types::CancelAllOrderIntent,
    tick: u64,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    use paraphina::live::types::{CancelAllAccepted, CancelAllRejected, ExecutionEvent};
    let mut events = Vec::new();
    let res = gateway
        .submit_intent(
            &paraphina::types::OrderIntent::CancelAll(cancel_all.clone()),
            tick,
            now_ms,
        )
        .await;
    match res {
        Ok(_resp) => {
            *seq = seq.wrapping_add(1);
            events.push(ExecutionEvent::CancelAllAccepted(CancelAllAccepted {
                venue_index: cancel_all.venue_index.unwrap_or(0),
                venue_id: cancel_all
                    .venue_id
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "all".to_string()),
                seq: *seq,
                timestamp_ms: now_ms,
                count: 0,
            }));
        }
        Err(err) => {
            *seq = seq.wrapping_add(1);
            events.push(ExecutionEvent::CancelAllRejected(CancelAllRejected {
                venue_index: cancel_all.venue_index.unwrap_or(0),
                venue_id: cancel_all
                    .venue_id
                    .as_ref()
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "all".to_string()),
                seq: *seq,
                timestamp_ms: now_ms,
                reason: err.message.clone(),
            }));
        }
    }
    events
}

fn send_unavailable_account_snapshot(
    account_tx: &mpsc::Sender<paraphina::live::types::AccountEvent>,
    cfg: &Config,
) {
    for (venue_index, venue) in cfg.venues.iter().enumerate() {
        let snapshot = paraphina::live::types::AccountSnapshot {
            venue_index,
            venue_id: venue.id.clone(),
            seq: 0,
            timestamp_ms: 0,
            positions: Vec::new(),
            balances: Vec::new(),
            funding_8h: None,
            margin: paraphina::live::types::MarginSnapshot {
                balance_usd: 0.0,
                used_usd: 0.0,
                available_usd: 0.0,
            },
            liquidation: paraphina::live::types::LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        };
        let _ = account_tx.try_send(paraphina::live::types::AccountEvent::Snapshot(snapshot));
    }
}

fn write_summary(
    out_dir: &std::path::Path,
    cfg: &Config,
    trade_mode: TradeMode,
    connector: &str,
    summary: &paraphina::live::LiveRunSummary,
    stats: Arc<Mutex<LiveTelemetryStats>>,
) {
    let stats = match stats.lock() {
        Ok(s) => s,
        Err(_) => return,
    };
    let fv_rate = if stats.ticks_total > 0 {
        stats.fv_available_ticks as f64 / stats.ticks_total as f64
    } else {
        0.0
    };
    let payload = serde_json::json!({
        "trade_mode": trade_mode.as_str(),
        "connector": connector,
        "venues": cfg.venues.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(),
        "ticks_run": summary.ticks_run,
        "run_duration_ms": summary.ticks_run as i64 * cfg.main_loop_interval_ms,
        "would_place_by_purpose": stats.would_place_by_purpose,
        "would_cancel_by_purpose": stats.would_cancel_by_purpose,
        "would_replace_by_purpose": stats.would_replace_by_purpose,
        "fv_available_rate": fv_rate,
        "venue_staleness_events": stats.venue_staleness_events,
        "venue_disabled_events": stats.venue_disabled_events,
        "kill_events": stats.kill_events,
    });
    let path = out_dir.join("summary.json");
    if let Ok(text) = serde_json::to_string_pretty(&payload) {
        let _ = std::fs::write(path, text);
    }
}

#[cfg(test)]
mod tests {
    use super::ConnectorArg;
    use paraphina::live::venues::ROADMAP_B_VENUES;

    #[test]
    fn roadmap_b_registry_is_complete() {
        assert_eq!(ROADMAP_B_VENUES.len(), 5);
        let selectable = ConnectorArg::roadmap_b_selectable_venues();
        assert_eq!(selectable, ROADMAP_B_VENUES.to_vec());
    }
}
