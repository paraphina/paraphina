//! Live trading skeleton binary (feature-gated).
//!
//! This binary wires the live cache, event model, and strategy loop together
//! without any external network connectors.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::ToSocketAddrs;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use paraphina::config::{resolve_effective_profile, Config};
use paraphina::io::GatewayPolicy;
use paraphina::live::gateway::{GatewayMux, LiveGateway, LiveRestClient};
use paraphina::live::instrument::{validate_specs, InstrumentSpec};
use paraphina::live::ops::{
    default_audit_dir, format_startup_log, start_metrics_server, write_audit_files,
    EnvSecretProvider, HealthState, LiveMetrics, SecretProvider,
};
use paraphina::live::orderbook_l2::BookLevel;
use paraphina::live::paper_adapter::{PaperExecutionAdapter, PaperFillMode, PaperMarketUpdate};
use paraphina::live::runner::{
    run_live_loop, LiveChannels, LiveOrderRequest, LiveRunMode, LiveRuntimeHooks,
};
use paraphina::live::shadow_adapter::ShadowAckAdapter;
use paraphina::live::types::L2Snapshot;
use paraphina::live::venues::{canonical_venue_ids, roadmap_b_enabled};
use paraphina::live::{resolve_effective_trade_mode, LiveTelemetry, LiveTelemetryStats, TradeMode};
use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink};
use url::Url;
use serde::Deserialize;
use std::path::PathBuf;

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

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq, Hash)]
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
            "hyperliquid_fixture"
            | "hyperliquid-fixture"
            | "hl_fixture"
            | "hl-fixture"
            | "fixture" => Some(ConnectorArg::HyperliquidFixture),
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
#[command(
    name = "paraphina_live",
    about = "Paraphina live runner (shadow-safe by default)",
    version
)]
struct Args {
    /// Trade mode: shadow (default), paper, testnet, live.
    #[arg(long, value_enum)]
    trade_mode: Option<TradeModeArg>,
    /// Connector to use: mock (default), hyperliquid, hyperliquid_fixture, lighter, extended, aster, paradex.
    #[arg(long, value_enum)]
    connector: Option<ConnectorArg>,
    /// Connectors to use (comma-separated list).
    #[arg(long)]
    connectors: Option<String>,
    /// Explicitly allow live execution (only applies to trade-mode=live).
    #[arg(long)]
    enable_live_execution: bool,
    /// Canary profile name or path (required for trade-mode=live).
    #[arg(long)]
    canary_profile: Option<String>,
    /// Run configuration checks and exit with PASS/FAIL status.
    #[arg(long)]
    preflight: bool,
    /// Output directory for telemetry/audit artifacts.
    #[arg(long)]
    out_dir: Option<String>,
    /// Force Extended to use fixture feed (disables live WS).
    #[arg(long)]
    extended_fixture: bool,
    /// Force Paradex to use fixture feed (disables live WS).
    #[arg(long)]
    paradex_fixture: bool,
    /// Force Aster to use fixture feed (disables live WS).
    #[arg(long)]
    aster_fixture: bool,
    /// Record live WS frames to fixtures dir (Aster/Extended/Paradex, manual runs).
    #[arg(long)]
    record_fixtures: bool,
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

#[derive(Debug, Clone)]
struct ConnectorSelection {
    connectors: Vec<ConnectorArg>,
    explicit_list: bool,
}

fn parse_connectors_list(raw: &str) -> Result<Vec<ConnectorArg>, String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut invalid = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        match ConnectorArg::parse_env(trimmed) {
            Some(connector) => {
                if seen.insert(connector) {
                    out.push(connector);
                }
            }
            None => invalid.push(trimmed.to_string()),
        }
    }
    if !invalid.is_empty() {
        return Err(format!("invalid connectors: {:?}", invalid));
    }
    if out.is_empty() {
        return Err("no connectors specified".to_string());
    }
    Ok(out)
}

fn resolve_connectors(args: &Args) -> ConnectorSelection {
    if let Some(raw) = args.connectors.as_ref() {
        if args.connector.is_some() {
            eprintln!("paraphina_live | warn=connector_ignored reason=connectors_list_set");
        }
        let connectors = parse_connectors_list(raw).unwrap_or_else(|err| {
            eprintln!("paraphina_live | error=invalid_connectors source=cli detail={err}");
            std::process::exit(2);
        });
        return ConnectorSelection {
            connectors,
            explicit_list: true,
        };
    }
    if let Ok(raw) = std::env::var("PARAPHINA_LIVE_CONNECTORS") {
        if !raw.trim().is_empty() {
            let connectors = parse_connectors_list(&raw).unwrap_or_else(|err| {
                eprintln!("paraphina_live | error=invalid_connectors source=env detail={err}");
                std::process::exit(2);
            });
            return ConnectorSelection {
                connectors,
                explicit_list: true,
            };
        }
    }
    ConnectorSelection {
        connectors: vec![resolve_connector(args.connector)],
        explicit_list: false,
    }
}

fn connector_venue_id(connector: ConnectorArg) -> &'static str {
    match connector {
        ConnectorArg::Hyperliquid | ConnectorArg::HyperliquidFixture => "hyperliquid",
        ConnectorArg::Lighter => "lighter",
        ConnectorArg::Extended => "extended",
        ConnectorArg::Aster => "aster",
        ConnectorArg::Paradex => "paradex",
        ConnectorArg::Mock => "mock",
    }
}

fn resolve_venue_index(cfg: &Config, venue_id: &str) -> Option<usize> {
    cfg.venues.iter().position(|venue| venue.id == venue_id)
}

fn resolve_connector_venue(
    cfg: &Config,
    connector: ConnectorArg,
) -> Result<(String, usize), String> {
    let venue_id = connector_venue_id(connector).to_string();
    if let Some(index) = resolve_venue_index(cfg, &venue_id) {
        return Ok((venue_id, index));
    }
    if connector == ConnectorArg::Mock {
        if let Some((index, venue)) = cfg.venues.iter().enumerate().next() {
            return Ok((venue.id.clone(), index));
        }
    }
    Err(format!(
        "connector_venue_missing connector={} venue_id={}",
        connector.as_str(),
        venue_id
    ))
}

fn resolve_fixture_dir(connector: ConnectorArg) -> Option<std::path::PathBuf> {
    let env_key = match connector {
        ConnectorArg::Paradex => "PARADEX_FIXTURE_DIR",
        ConnectorArg::Aster => "ASTER_FIXTURE_DIR",
        ConnectorArg::Extended => "EXTENDED_FIXTURE_DIR",
        _ => return None,
    };
    if let Ok(val) = std::env::var(env_key) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return Some(std::path::PathBuf::from(trimmed));
        }
    }
    if let Ok(root) = std::env::var("ROADMAP_B_FIXTURE_DIR") {
        let trimmed = root.trim();
        if !trimmed.is_empty() {
            return Some(std::path::PathBuf::from(trimmed).join(connector_venue_id(connector)));
        }
    }
    None
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

#[derive(Debug, Deserialize)]
struct CanaryVenueConfig {
    base_order_size: Option<f64>,
    max_order_size: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CanaryLimitsConfig {
    max_position_tao: Option<f64>,
    max_open_orders: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CanaryRateLimitConfig {
    enabled: Option<bool>,
    rps: Option<f64>,
    burst: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct CanaryEnforcementConfig {
    post_only: Option<bool>,
    reduce_only: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CanaryKillConfig {
    stale_max_ticks: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct CanaryConfig {
    venue: Option<CanaryVenueConfig>,
    limits: Option<CanaryLimitsConfig>,
    rate_limit: Option<CanaryRateLimitConfig>,
    enforcement: Option<CanaryEnforcementConfig>,
    kill: Option<CanaryKillConfig>,
}

#[derive(Debug, Default)]
struct CanarySettings {
    max_position_tao: Option<f64>,
    max_open_orders: Option<usize>,
    stale_max_ticks: Option<u64>,
    enforce_post_only: bool,
    enforce_reduce_only: bool,
    rate_limit_enabled: Option<bool>,
    rate_limit_rps: Option<f64>,
    rate_limit_burst: Option<u32>,
}

fn resolve_canary_profile(cli: Option<String>) -> Option<PathBuf> {
    let val = cli.or_else(|| std::env::var("PARAPHINA_LIVE_CANARY_PROFILE").ok())?;
    let trimmed = val.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "prod_canary" {
        return Some(PathBuf::from("configs").join("prod_canary.toml"));
    }
    Some(PathBuf::from(trimmed))
}

fn load_canary_config(path: &PathBuf) -> Result<CanaryConfig, String> {
    let raw = std::fs::read_to_string(path).map_err(|err| {
        format!(
            "canary_profile_read_error path={} err={}",
            path.display(),
            err
        )
    })?;
    toml::from_str::<CanaryConfig>(&raw).map_err(|err| {
        format!(
            "canary_profile_parse_error path={} err={}",
            path.display(),
            err
        )
    })
}

fn apply_canary_config(cfg: &mut Config, canary: &CanaryConfig) -> CanarySettings {
    let mut settings = CanarySettings::default();
    if let Some(venue) = &canary.venue {
        for v in &mut cfg.venues {
            if let Some(size) = venue.base_order_size {
                v.base_order_size = size.max(0.0);
            }
            if let Some(size) = venue.max_order_size {
                v.max_order_size = size.max(0.0);
            }
        }
    }
    if let Some(limits) = &canary.limits {
        settings.max_position_tao = limits.max_position_tao;
        settings.max_open_orders = limits.max_open_orders;
    }
    if let Some(rate) = &canary.rate_limit {
        settings.rate_limit_enabled = rate.enabled;
        settings.rate_limit_rps = rate.rps;
        settings.rate_limit_burst = rate.burst;
    }
    if let Some(enforce) = &canary.enforcement {
        settings.enforce_post_only = enforce.post_only.unwrap_or(false);
        settings.enforce_reduce_only = enforce.reduce_only.unwrap_or(false);
    }
    if let Some(kill) = &canary.kill {
        settings.stale_max_ticks = kill.stale_max_ticks;
    }
    settings
}

fn apply_canary_env(settings: &CanarySettings) {
    std::env::set_var("PARAPHINA_CANARY_MODE", "1");
    if let Some(val) = settings.max_position_tao {
        std::env::set_var("PARAPHINA_CANARY_MAX_POSITION_TAO", val.to_string());
    }
    if let Some(val) = settings.max_open_orders {
        std::env::set_var("PARAPHINA_CANARY_MAX_OPEN_ORDERS", val.to_string());
    }
    if let Some(val) = settings.stale_max_ticks {
        std::env::set_var("PARAPHINA_CANARY_STALE_MAX_TICKS", val.to_string());
    }
    std::env::set_var(
        "PARAPHINA_CANARY_ENFORCE_POST_ONLY",
        if settings.enforce_post_only { "1" } else { "0" },
    );
    std::env::set_var(
        "PARAPHINA_CANARY_ENFORCE_REDUCE_ONLY",
        if settings.enforce_reduce_only {
            "1"
        } else {
            "0"
        },
    );
    if let Some(val) = settings.rate_limit_enabled {
        std::env::set_var("PARAPHINA_RATE_LIMIT_ENABLED", if val { "1" } else { "0" });
    }
    if let Some(val) = settings.rate_limit_rps {
        std::env::set_var("PARAPHINA_RATE_LIMIT_RPS", val.to_string());
    }
    if let Some(val) = settings.rate_limit_burst {
        std::env::set_var("PARAPHINA_RATE_LIMIT_BURST", val.to_string());
    }
}

fn resolve_canary_settings(
    trade_mode: TradeMode,
    cfg: &mut Config,
    canary_profile: Option<&PathBuf>,
    apply_env: bool,
) -> Result<Option<CanarySettings>, String> {
    if trade_mode != TradeMode::Live {
        return Ok(None);
    }
    let Some(path) = canary_profile else {
        return Err("canary profile not set".to_string());
    };
    let canary = load_canary_config(path)?;
    let settings = apply_canary_config(cfg, &canary);
    if apply_env {
        apply_canary_env(&settings);
    }
    Ok(Some(settings))
}

fn resolve_telemetry_path(out_dir: Option<&std::path::PathBuf>) -> Option<std::path::PathBuf> {
    let mut telemetry_path = std::env::var("PARAPHINA_TELEMETRY_PATH")
        .ok()
        .map(std::path::PathBuf::from);
    if telemetry_path.is_none() {
        if let Some(out_dir) = out_dir {
            telemetry_path = Some(out_dir.join("telemetry.jsonl"));
        }
    }
    telemetry_path
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

fn env_is_true(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

fn env_is_yes(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v.trim().eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

fn env_present(name: &str) -> bool {
    std::env::var(name)
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
struct ReconcileEnvState {
    enabled: bool,
    detail: String,
}

fn parse_reconcile_env() -> ReconcileEnvState {
    let raw = match std::env::var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS") {
        Ok(val) => val,
        Err(_) => {
            return ReconcileEnvState {
                enabled: false,
                detail: "missing".to_string(),
            }
        }
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "false" | "off" | "no") {
        return ReconcileEnvState {
            enabled: false,
            detail: "disabled (explicit)".to_string(),
        };
    }
    if let Ok(ms) = raw.parse::<i64>() {
        if ms > 0 {
            return ReconcileEnvState {
                enabled: true,
                detail: format!("enabled ms={}", ms),
            };
        }
        return ReconcileEnvState {
            enabled: false,
            detail: format!("disabled value={}", ms),
        };
    }
    ReconcileEnvState {
        enabled: false,
        detail: format!("invalid value={}", raw),
    }
}

fn endpoint_dns_status(url: &str) -> (bool, String) {
    let parsed = match Url::parse(url) {
        Ok(parsed) => parsed,
        Err(err) => return (false, format!("invalid_url err={err} url={url}")),
    };
    let Some(host) = parsed.host_str() else {
        return (false, format!("invalid_host url={url}"));
    };
    let Some(port) = parsed.port_or_known_default() else {
        return (false, format!("invalid_port url={url}"));
    };
    let addr = format!("{host}:{port}");
    match addr.to_socket_addrs() {
        Ok(mut resolved) => {
            if resolved.next().is_some() {
                (true, format!("dns_ok host={host} port={port}"))
            } else {
                (false, format!("dns_empty host={host} port={port}"))
            }
        }
        Err(err) => (false, format!("dns_fail host={host} port={port} err={err}")),
    }
}

fn append_endpoint_details(
    endpoint_details: &mut Vec<String>,
    endpoint_ok: &mut bool,
    label: &str,
    ws_url: Option<&str>,
    http_url: Option<&str>,
) {
    let mut parts = Vec::new();
    if let Some(ws) = ws_url {
        let (ok, status) = endpoint_dns_status(ws);
        *endpoint_ok &= ok;
        parts.push(format!("ws={ws} {status}"));
    }
    if let Some(http) = http_url {
        let (ok, status) = endpoint_dns_status(http);
        *endpoint_ok &= ok;
        parts.push(format!("http={http} {status}"));
    }
    if parts.is_empty() {
        endpoint_details.push(format!("{label}:n/a"));
    } else {
        endpoint_details.push(format!("{label} {}", parts.join(" ")));
    }
}

fn paradex_fixture_mode(args: &Args) -> bool {
    args.paradex_fixture || env_is_true("PARADEX_FIXTURE_MODE")
}

fn aster_fixture_mode(args: &Args) -> bool {
    args.aster_fixture || env_is_true("ASTER_FIXTURE_MODE")
}

fn extended_fixture_mode(args: &Args) -> bool {
    args.extended_fixture || env_is_true("EXTENDED_FIXTURE_MODE")
}

fn paradex_record_enabled(args: &Args) -> bool {
    args.record_fixtures || env_is_true("PARADEX_RECORD_FIXTURES")
}

fn aster_record_enabled(args: &Args) -> bool {
    args.record_fixtures || env_is_true("ASTER_RECORD_FIXTURES")
}

fn extended_record_enabled(args: &Args) -> bool {
    args.record_fixtures || env_is_true("EXTENDED_RECORD_FIXTURES")
}

fn resolve_paradex_record_dir() -> PathBuf {
    std::env::var("PARADEX_RECORD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./tests/fixtures/roadmap_b/paradex_live_recording"))
}

fn resolve_aster_record_dir() -> PathBuf {
    std::env::var("ASTER_RECORD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./tests/fixtures/roadmap_b/aster_live_recording"))
}

fn resolve_extended_record_dir() -> PathBuf {
    std::env::var("EXTENDED_RECORD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./tests/fixtures/roadmap_b/extended_live_recording"))
}

fn aster_ws_url() -> String {
    std::env::var("ASTER_WS_URL").unwrap_or_else(|_| "wss://fstream.asterdex.com/ws".to_string())
}

fn extended_ws_url() -> String {
    std::env::var("EXTENDED_WS_URL")
        .unwrap_or_else(|_| "wss://stream.extended.exchange/ws".to_string())
}

fn paradex_ws_url() -> String {
    std::env::var("PARADEX_WS_URL")
        .unwrap_or_else(|_| "wss://ws.api.prod.paradex.trade/v1".to_string())
}

fn aster_market_symbol() -> String {
    std::env::var("ASTER_MARKET").unwrap_or_else(|_| "BTCUSDT".to_string())
}

fn extended_market_symbol() -> String {
    std::env::var("EXTENDED_MARKET").unwrap_or_else(|_| "BTCUSDT".to_string())
}

fn paradex_market_symbol() -> String {
    std::env::var("PARADEX_MARKET").unwrap_or_else(|_| "BTC-USD-PERP".to_string())
}

fn is_valid_ws_url(url: &str) -> bool {
    url.starts_with("wss://") || url.starts_with("ws://")
}

fn is_valid_symbol(symbol: &str) -> bool {
    !symbol.trim().is_empty()
        && symbol
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn live_connector_allowed_for_live_mode(connector: ConnectorArg) -> bool {
    matches!(
        connector,
        ConnectorArg::Hyperliquid
            | ConnectorArg::Lighter
            | ConnectorArg::Aster
            | ConnectorArg::Extended
            | ConnectorArg::Paradex
    )
}

fn connectors_allowed_for_live_mode(connectors: &[ConnectorArg]) -> bool {
    connectors
        .iter()
        .copied()
        .all(live_connector_allowed_for_live_mode)
}

fn connectors_label(connectors: &[ConnectorArg]) -> String {
    connectors
        .iter()
        .map(ConnectorArg::as_str)
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConnectorSupport {
    MissingFeature,
    Stub,
    MarketOnly,
    MarketAccount,
    MarketAccountExec,
}

fn connector_support(connector: ConnectorArg) -> ConnectorSupport {
    match connector {
        ConnectorArg::Mock => ConnectorSupport::MarketOnly,
        ConnectorArg::Hyperliquid => ConnectorSupport::MarketAccountExec,
        ConnectorArg::HyperliquidFixture => ConnectorSupport::MarketAccount,
        ConnectorArg::Lighter => ConnectorSupport::MarketAccountExec,
        ConnectorArg::Extended => {
            if cfg!(feature = "live_extended") {
                ConnectorSupport::MarketAccountExec
            } else {
                ConnectorSupport::MissingFeature
            }
        }
        ConnectorArg::Aster => {
            if cfg!(feature = "live_aster") {
                ConnectorSupport::MarketAccountExec
            } else {
                ConnectorSupport::MissingFeature
            }
        }
        ConnectorArg::Paradex => {
            if cfg!(feature = "live_paradex") {
                ConnectorSupport::MarketAccountExec
            } else {
                ConnectorSupport::MissingFeature
            }
        }
    }
}

fn paper_market_update_from_event(
    event: &paraphina::live::types::MarketDataEvent,
) -> Option<PaperMarketUpdate> {
    match event {
        paraphina::live::types::MarketDataEvent::L2Snapshot(snapshot) => {
            let best_bid = snapshot.bids.first().map(|level| level.price);
            let best_ask = snapshot.asks.first().map(|level| level.price);
            Some(PaperMarketUpdate {
                venue_index: snapshot.venue_index,
                best_bid,
                best_ask,
                timestamp_ms: snapshot.timestamp_ms,
            })
        }
        _ => None,
    }
}

fn override_market_timestamp(
    event: paraphina::live::types::MarketDataEvent,
    timestamp_ms: i64,
) -> paraphina::live::types::MarketDataEvent {
    match event {
        paraphina::live::types::MarketDataEvent::L2Snapshot(mut snapshot) => {
            snapshot.timestamp_ms = timestamp_ms;
            paraphina::live::types::MarketDataEvent::L2Snapshot(snapshot)
        }
        paraphina::live::types::MarketDataEvent::L2Delta(mut delta) => {
            delta.timestamp_ms = timestamp_ms;
            paraphina::live::types::MarketDataEvent::L2Delta(delta)
        }
        paraphina::live::types::MarketDataEvent::Trade(mut trade) => {
            trade.timestamp_ms = timestamp_ms;
            paraphina::live::types::MarketDataEvent::Trade(trade)
        }
        paraphina::live::types::MarketDataEvent::FundingUpdate(mut update) => {
            update.timestamp_ms = timestamp_ms;
            paraphina::live::types::MarketDataEvent::FundingUpdate(update)
        }
    }
}

fn rewrite_market_event(
    event: paraphina::live::types::MarketDataEvent,
    venue_id: &str,
    venue_index: usize,
) -> paraphina::live::types::MarketDataEvent {
    match event {
        paraphina::live::types::MarketDataEvent::L2Snapshot(mut snapshot) => {
            snapshot.venue_id = venue_id.to_string();
            snapshot.venue_index = venue_index;
            paraphina::live::types::MarketDataEvent::L2Snapshot(snapshot)
        }
        paraphina::live::types::MarketDataEvent::L2Delta(mut delta) => {
            delta.venue_id = venue_id.to_string();
            delta.venue_index = venue_index;
            paraphina::live::types::MarketDataEvent::L2Delta(delta)
        }
        paraphina::live::types::MarketDataEvent::Trade(mut trade) => {
            trade.venue_id = venue_id.to_string();
            trade.venue_index = venue_index;
            paraphina::live::types::MarketDataEvent::Trade(trade)
        }
        paraphina::live::types::MarketDataEvent::FundingUpdate(mut update) => {
            update.venue_id = venue_id.to_string();
            update.venue_index = venue_index;
            paraphina::live::types::MarketDataEvent::FundingUpdate(update)
        }
    }
}

fn rewrite_account_event(
    event: paraphina::live::types::AccountEvent,
    venue_id: &str,
    venue_index: usize,
) -> paraphina::live::types::AccountEvent {
    match event {
        paraphina::live::types::AccountEvent::Snapshot(mut snapshot) => {
            snapshot.venue_id = venue_id.to_string();
            snapshot.venue_index = venue_index;
            paraphina::live::types::AccountEvent::Snapshot(snapshot)
        }
    }
}

fn rewrite_execution_event(
    event: paraphina::live::types::ExecutionEvent,
    venue_id: &str,
    venue_index: usize,
) -> paraphina::live::types::ExecutionEvent {
    match event {
        paraphina::live::types::ExecutionEvent::OrderAccepted(mut ack) => {
            ack.venue_id = venue_id.to_string();
            ack.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::OrderAccepted(ack)
        }
        paraphina::live::types::ExecutionEvent::OrderRejected(mut rej) => {
            rej.venue_id = venue_id.to_string();
            rej.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::OrderRejected(rej)
        }
        paraphina::live::types::ExecutionEvent::Filled(mut fill) => {
            fill.venue_id = venue_id.to_string();
            fill.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::Filled(fill)
        }
        paraphina::live::types::ExecutionEvent::CancelAccepted(mut ack) => {
            ack.venue_id = venue_id.to_string();
            ack.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::CancelAccepted(ack)
        }
        paraphina::live::types::ExecutionEvent::CancelRejected(mut rej) => {
            rej.venue_id = venue_id.to_string();
            rej.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::CancelRejected(rej)
        }
        paraphina::live::types::ExecutionEvent::CancelAllAccepted(mut ack) => {
            ack.venue_id = venue_id.to_string();
            ack.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::CancelAllAccepted(ack)
        }
        paraphina::live::types::ExecutionEvent::CancelAllRejected(mut rej) => {
            rej.venue_id = venue_id.to_string();
            rej.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::CancelAllRejected(rej)
        }
        paraphina::live::types::ExecutionEvent::OrderSnapshot(mut snapshot) => {
            snapshot.venue_id = venue_id.to_string();
            snapshot.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::OrderSnapshot(snapshot)
        }
    }
}

struct ConnectorChannels {
    market_tx: mpsc::Sender<paraphina::live::types::MarketDataEvent>,
    account_tx: mpsc::Sender<paraphina::live::types::AccountEvent>,
    exec_tx: mpsc::Sender<paraphina::live::types::ExecutionEvent>,
}

fn spawn_connector_forwarders(
    venue_id: String,
    venue_index: usize,
    market_rx: mpsc::Receiver<paraphina::live::types::MarketDataEvent>,
    account_rx: mpsc::Receiver<paraphina::live::types::AccountEvent>,
    exec_rx: mpsc::Receiver<paraphina::live::types::ExecutionEvent>,
    market_ingest_tx: mpsc::Sender<paraphina::live::types::MarketDataEvent>,
    account_tx: mpsc::Sender<paraphina::live::types::AccountEvent>,
    exec_tx: mpsc::Sender<paraphina::live::types::ExecutionEvent>,
) {
    let venue_id_market = venue_id.clone();
    let venue_id_account = venue_id.clone();
    let venue_id_exec = venue_id.clone();
    tokio::spawn(async move {
        let mut rx = market_rx;
        while let Some(event) = rx.recv().await {
            let event = rewrite_market_event(event, &venue_id_market, venue_index);
            let _ = market_ingest_tx.send(event).await;
        }
    });
    tokio::spawn(async move {
        let mut rx = account_rx;
        while let Some(event) = rx.recv().await {
            let event = rewrite_account_event(event, &venue_id_account, venue_index);
            let _ = account_tx.send(event).await;
        }
    });
    tokio::spawn(async move {
        let mut rx = exec_rx;
        while let Some(event) = rx.recv().await {
            let event = rewrite_execution_event(event, &venue_id_exec, venue_index);
            let _ = exec_tx.send(event).await;
        }
    });
}

fn enforce_live_execution_guardrails(
    args: &Args,
    trade_mode: TradeMode,
    connectors: &[ConnectorArg],
    canary_profile: Option<&PathBuf>,
    canary_settings: Option<&CanarySettings>,
) {
    if trade_mode != TradeMode::Live {
        return;
    }

    let exec_enable_env = env_is_true("PARAPHINA_LIVE_EXEC_ENABLE");
    let exec_confirm_env = env_is_yes("PARAPHINA_LIVE_EXECUTION_CONFIRM");
    let live_flag = args.enable_live_execution;

    if !connectors_allowed_for_live_mode(connectors) {
        eprintln!(
            "paraphina_live | error=live_mode_connector_invalid connectors={} (use --trade-mode shadow for safe runs)",
            connectors_label(connectors)
        );
        std::process::exit(2);
    }

    let preflight_ok = env_is_true("PARAPHINA_LIVE_PREFLIGHT_OK");
    if !preflight_ok {
        eprintln!(
            "paraphina_live | error=live_mode_preflight_missing (set PARAPHINA_LIVE_PREFLIGHT_OK=1 after preflight)"
        );
        std::process::exit(2);
    }
    let reconcile_state = parse_reconcile_env();
    if !reconcile_state.enabled {
        eprintln!(
            "paraphina_live | error=live_mode_reconcile_missing (set PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS)"
        );
        std::process::exit(2);
    }
    if canary_profile.is_none() {
        eprintln!(
            "paraphina_live | error=live_mode_canary_profile_missing (set --canary-profile or PARAPHINA_LIVE_CANARY_PROFILE)"
        );
        std::process::exit(2);
    }
    if let Some(settings) = canary_settings {
        if settings.max_position_tao.is_none() || settings.max_open_orders.is_none() {
            eprintln!(
                "paraphina_live | error=live_mode_canary_caps_missing (max_position_tao and max_open_orders required)"
            );
            std::process::exit(2);
        }
    }
    if !live_flag || !exec_enable_env || !exec_confirm_env {
        eprintln!(
            "paraphina_live | error=live_mode_guardrails_missing enable_flag={} exec_env={} confirm_env={} (use --trade-mode shadow for safe runs)",
            live_flag, exec_enable_env, exec_confirm_env
        );
        std::process::exit(2);
    }
}

struct PreflightCheck {
    label: &'static str,
    ok: bool,
    details: String,
}

fn run_preflight(
    args: &Args,
    trade_mode: TradeMode,
    connectors: &[ConnectorArg],
    cfg: &Config,
    out_dir: Option<std::path::PathBuf>,
    canary_error: Option<&str>,
    canary_settings: Option<&CanarySettings>,
) -> bool {
    let mut checks: Vec<PreflightCheck> = Vec::new();

    let trade_mode_detail = format!("selected={}", trade_mode.as_str());
    checks.push(PreflightCheck {
        label: "trade_mode",
        ok: true,
        details: trade_mode_detail,
    });

    let mut connector_ok = true;
    let mut connector_details = Vec::new();
    for connector in connectors {
        let support = connector_support(*connector);
        let supported = matches!(
            support,
            ConnectorSupport::MarketOnly
                | ConnectorSupport::MarketAccount
                | ConnectorSupport::MarketAccountExec
        );
        if !supported {
            connector_ok = false;
        }
        connector_details.push(format!("{}:{:?}", connector.as_str(), support));
    }
    checks.push(PreflightCheck {
        label: "connectors",
        ok: connector_ok,
        details: connector_details.join(","),
    });

    let mut endpoint_ok = true;
    let mut endpoint_details = Vec::new();
    for connector in connectors {
        match connector {
            ConnectorArg::Mock => {
                append_endpoint_details(&mut endpoint_details, &mut endpoint_ok, "mock", None, None)
            }
            ConnectorArg::HyperliquidFixture => {
                endpoint_details.push("hyperliquid_fixture:fixture_mode".to_string());
            }
            ConnectorArg::Hyperliquid => {
                #[cfg(feature = "live_hyperliquid")]
                {
                    let cfg =
                        paraphina::live::connectors::hyperliquid::HyperliquidConfig::from_env();
                    append_endpoint_details(
                        &mut endpoint_details,
                        &mut endpoint_ok,
                        "hyperliquid",
                        Some(cfg.ws_url.as_str()),
                        Some(cfg.rest_url.as_str()),
                    );
                    let (ok, status) = endpoint_dns_status(cfg.info_url.as_str());
                    endpoint_ok &= ok;
                    endpoint_details
                        .push(format!("hyperliquid_info http={} {status}", cfg.info_url));
                }
                #[cfg(not(feature = "live_hyperliquid"))]
                {
                    endpoint_ok = false;
                    endpoint_details.push("hyperliquid:feature_disabled".to_string());
                }
            }
            ConnectorArg::Lighter => {
                #[cfg(feature = "live_lighter")]
                {
                    let cfg = paraphina::live::connectors::lighter::LighterConfig::from_env();
                    append_endpoint_details(
                        &mut endpoint_details,
                        &mut endpoint_ok,
                        "lighter",
                        Some(cfg.ws_url.as_str()),
                        Some(cfg.rest_url.as_str()),
                    );
                }
                #[cfg(not(feature = "live_lighter"))]
                {
                    endpoint_ok = false;
                    endpoint_details.push("lighter:feature_disabled".to_string());
                }
            }
            ConnectorArg::Extended => {
                #[cfg(feature = "live_extended")]
                {
                    let cfg = paraphina::live::connectors::extended::ExtendedConfig::from_env();
                    append_endpoint_details(
                        &mut endpoint_details,
                        &mut endpoint_ok,
                        "extended",
                        Some(cfg.ws_url.as_str()),
                        Some(cfg.rest_url.as_str()),
                    );
                }
                #[cfg(not(feature = "live_extended"))]
                {
                    endpoint_ok = false;
                    endpoint_details.push("extended:feature_disabled".to_string());
                }
            }
            ConnectorArg::Aster => {
                #[cfg(feature = "live_aster")]
                {
                    let cfg = paraphina::live::connectors::aster::AsterConfig::from_env();
                    append_endpoint_details(
                        &mut endpoint_details,
                        &mut endpoint_ok,
                        "aster",
                        Some(cfg.ws_url.as_str()),
                        Some(cfg.rest_url.as_str()),
                    );
                }
                #[cfg(not(feature = "live_aster"))]
                {
                    endpoint_ok = false;
                    endpoint_details.push("aster:feature_disabled".to_string());
                }
            }
            ConnectorArg::Paradex => {
                #[cfg(feature = "live_paradex")]
                {
                    let cfg = paraphina::live::connectors::paradex::ParadexConfig::from_env();
                    append_endpoint_details(
                        &mut endpoint_details,
                        &mut endpoint_ok,
                        "paradex",
                        Some(cfg.ws_url.as_str()),
                        Some(cfg.rest_url.as_str()),
                    );
                }
                #[cfg(not(feature = "live_paradex"))]
                {
                    endpoint_ok = false;
                    endpoint_details.push("paradex:feature_disabled".to_string());
                }
            }
        }
    }
    checks.push(PreflightCheck {
        label: "connector_endpoints",
        ok: endpoint_ok,
        details: endpoint_details.join(" | "),
    });

    let mut venue_ok = true;
    let mut venue_details = Vec::new();
    for connector in connectors {
        match resolve_connector_venue(cfg, *connector) {
            Ok((venue_id, index)) => {
                venue_details.push(format!("{}:{}", venue_id, index));
            }
            Err(err) => {
                venue_ok = false;
                venue_details.push(err);
            }
        }
    }
    checks.push(PreflightCheck {
        label: "venues",
        ok: venue_ok,
        details: venue_details.join(","),
    });

    if connectors.iter().any(|c| *c == ConnectorArg::Extended) && !extended_fixture_mode(args) {
        let ws_url = extended_ws_url();
        let market = extended_market_symbol();
        let ws_ok = is_valid_ws_url(&ws_url);
        let market_ok = is_valid_symbol(&market);
        checks.push(PreflightCheck {
            label: "extended_ws_url",
            ok: ws_ok,
            details: if ws_ok {
                "ok".to_string()
            } else {
                format!("invalid url={}", ws_url)
            },
        });
        checks.push(PreflightCheck {
            label: "extended_market",
            ok: market_ok,
            details: if market_ok {
                format!("symbol={}", market)
            } else {
                format!("invalid symbol={}", market)
            },
        });
    }

    if connectors.iter().any(|c| *c == ConnectorArg::Paradex) && !paradex_fixture_mode(args) {
        let ws_url = paradex_ws_url();
        let market = paradex_market_symbol();
        let ws_ok = is_valid_ws_url(&ws_url);
        let market_ok = is_valid_symbol(&market);
        checks.push(PreflightCheck {
            label: "paradex_ws_url",
            ok: ws_ok,
            details: if ws_ok {
                "ok".to_string()
            } else {
                format!("invalid url={}", ws_url)
            },
        });
        checks.push(PreflightCheck {
            label: "paradex_market",
            ok: market_ok,
            details: if market_ok {
                format!("symbol={}", market)
            } else {
                format!("invalid symbol={}", market)
            },
        });
    }

    if connectors.iter().any(|c| *c == ConnectorArg::Aster) && !aster_fixture_mode(args) {
        let ws_url = aster_ws_url();
        let market = aster_market_symbol();
        let ws_ok = is_valid_ws_url(&ws_url);
        let market_ok = is_valid_symbol(&market);
        checks.push(PreflightCheck {
            label: "aster_ws_url",
            ok: ws_ok,
            details: if ws_ok {
                "ok".to_string()
            } else {
                format!("invalid url={}", ws_url)
            },
        });
        checks.push(PreflightCheck {
            label: "aster_market",
            ok: market_ok,
            details: if market_ok {
                format!("symbol={}", market)
            } else {
                format!("invalid symbol={}", market)
            },
        });
    }

    if trade_mode == TradeMode::Live {
        let canary_ok = canary_error.is_none()
            && canary_settings
                .map(|settings| {
                    settings.max_position_tao.is_some() && settings.max_open_orders.is_some()
                })
                .unwrap_or(false);
        checks.push(PreflightCheck {
            label: "canary_profile",
            ok: canary_ok,
            details: canary_error.unwrap_or("loaded").to_string(),
        });
        let reconcile_state = parse_reconcile_env();
        checks.push(PreflightCheck {
            label: "reconciliation",
            ok: reconcile_state.enabled,
            details: reconcile_state.detail.clone(),
        });
    }

    let audit_dir = out_dir.clone().unwrap_or_else(default_audit_dir);
    let out_dir_ok = std::fs::create_dir_all(&audit_dir).is_ok()
        && std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(audit_dir.join(".preflight_write_test"))
            .is_ok();
    let _ = std::fs::remove_file(audit_dir.join(".preflight_write_test"));
    checks.push(PreflightCheck {
        label: "out_dir",
        ok: out_dir_ok,
        details: format!("path={}", audit_dir.display()),
    });

    let telemetry_path = resolve_telemetry_path(out_dir.as_ref());
    let telemetry_mode = TelemetryMode::from_env();
    let telemetry_ok = match telemetry_mode {
        TelemetryMode::Off => true,
        TelemetryMode::Jsonl => telemetry_path.as_ref().is_some_and(|path| {
            path.parent()
                .map(|p| std::fs::create_dir_all(p).is_ok())
                .unwrap_or(true)
                && std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .open(path)
                    .is_ok()
        }),
    };
    checks.push(PreflightCheck {
        label: "telemetry",
        ok: telemetry_ok,
        details: match telemetry_mode {
            TelemetryMode::Off => "mode=off".to_string(),
            TelemetryMode::Jsonl => format!(
                "mode=jsonl path={}",
                telemetry_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<missing>".to_string())
            ),
        },
    });

    let mut creds_ok = true;
    let mut creds_detail = String::new();
    for connector in connectors {
        match connector {
            ConnectorArg::Hyperliquid | ConnectorArg::HyperliquidFixture => {
                let key_present = env_present("HL_PRIVATE_KEY");
                let vault_present = env_present("HL_VAULT_ADDRESS");
                if trade_mode == TradeMode::Live {
                    creds_ok = creds_ok && key_present && vault_present;
                }
                let mut detail = format!(
                    "{}:hl_private_key={} hl_vault_address={}",
                    connector.as_str(),
                    key_present,
                    vault_present
                );
                if *connector == ConnectorArg::HyperliquidFixture {
                    let fixture_dir = std::env::var("HL_FIXTURE_DIR")
                        .map(std::path::PathBuf::from)
                        .unwrap_or_else(|_| {
                            std::path::PathBuf::from("./tests/fixtures/hyperliquid")
                        });
                    let fixture_ok = fixture_dir.is_dir();
                    creds_ok = creds_ok && fixture_ok;
                    detail.push_str(&format!(" fixture_dir_ok={}", fixture_ok));
                }
                creds_detail.push_str(&detail);
            }
            ConnectorArg::Lighter => {
                let fixture_dir = std::env::var("LIGHTER_FIXTURE_DIR").ok();
                let fixture_ok = fixture_dir
                    .as_ref()
                    .map(|dir| std::path::Path::new(dir).is_dir())
                    .unwrap_or(false);
                if trade_mode == TradeMode::Live {
                    creds_ok = creds_ok && true;
                }
                let detail = format!(
                    "{}:no_key_required=true fixture_dir_ok={}",
                    connector.as_str(),
                    fixture_ok
                );
                creds_detail.push_str(&detail);
            }
            ConnectorArg::Mock => {
                creds_ok = creds_ok && trade_mode != TradeMode::Live;
                let detail = format!("{}:no_live_keys", connector.as_str());
                creds_detail.push_str(&detail);
            }
            ConnectorArg::Paradex => {
                let use_fixture = paradex_fixture_mode(args);
                if use_fixture {
                    let fixture_dir = resolve_fixture_dir(*connector);
                    let fixture_ok = fixture_dir
                        .as_ref()
                        .map(|dir| dir.is_dir())
                        .unwrap_or(false);
                    creds_ok = creds_ok && fixture_ok;
                    let detail = format!(
                        "{}:fixture_dir_ok={} fixture_mode=true",
                        connector.as_str(),
                        fixture_ok
                    );
                    creds_detail.push_str(&detail);
                } else {
                    let ws_ok = is_valid_ws_url(&paradex_ws_url());
                    let market_ok = is_valid_symbol(&paradex_market_symbol());
                    let jwt_present = env_present("PARADEX_JWT");
                    let payload_present = env_present("PARADEX_AUTH_PAYLOAD_JSON");
                    let needs_auth = matches!(trade_mode, TradeMode::Live | TradeMode::Testnet);
                    if needs_auth {
                        creds_ok = creds_ok && (jwt_present || payload_present);
                    }
                    creds_ok = creds_ok && ws_ok && market_ok;
                    let detail = format!(
                        "{}:public_ws=true ws_url_ok={} market_ok={} jwt={} auth_payload={}",
                        connector.as_str(),
                        ws_ok,
                        market_ok,
                        jwt_present,
                        payload_present
                    );
                    creds_detail.push_str(&detail);
                }
            }
            ConnectorArg::Extended => {
                let use_fixture = extended_fixture_mode(args);
                if use_fixture {
                    let fixture_dir = resolve_fixture_dir(*connector);
                    let fixture_ok = fixture_dir
                        .as_ref()
                        .map(|dir| dir.is_dir())
                        .unwrap_or(false);
                    creds_ok = creds_ok && fixture_ok;
                    let detail = format!(
                        "{}:fixture_dir_ok={} fixture_mode=true",
                        connector.as_str(),
                        fixture_ok
                    );
                    creds_detail.push_str(&detail);
                } else {
                    let ws_ok = is_valid_ws_url(&extended_ws_url());
                    let market_ok = is_valid_symbol(&extended_market_symbol());
                    let key_present = env_present("EXTENDED_API_KEY");
                    let secret_present = env_present("EXTENDED_API_SECRET");
                    let needs_keys = matches!(trade_mode, TradeMode::Live | TradeMode::Testnet);
                    if needs_keys {
                        creds_ok = creds_ok && key_present && secret_present;
                    }
                    creds_ok = creds_ok && ws_ok && market_ok;
                    let detail = format!(
                        "{}:public_ws=true ws_url_ok={} market_ok={} api_key={} api_secret={}",
                        connector.as_str(),
                        ws_ok,
                        market_ok,
                        key_present,
                        secret_present
                    );
                    creds_detail.push_str(&detail);
                }
            }
            ConnectorArg::Aster => {
                let use_fixture = aster_fixture_mode(args);
                if use_fixture {
                    let fixture_dir = resolve_fixture_dir(*connector);
                    let fixture_ok = fixture_dir
                        .as_ref()
                        .map(|dir| dir.is_dir())
                        .unwrap_or(false);
                    creds_ok = creds_ok && fixture_ok;
                    let detail = format!(
                        "{}:fixture_dir_ok={} fixture_mode=true",
                        connector.as_str(),
                        fixture_ok
                    );
                    creds_detail.push_str(&detail);
                } else {
                    let ws_ok = is_valid_ws_url(&aster_ws_url());
                    let market_ok = is_valid_symbol(&aster_market_symbol());
                    let key_present = env_present("ASTER_API_KEY");
                    let secret_present = env_present("ASTER_API_SECRET");
                    let needs_keys = matches!(trade_mode, TradeMode::Live | TradeMode::Testnet);
                    if needs_keys {
                        creds_ok = creds_ok && key_present && secret_present;
                    }
                    creds_ok = creds_ok && ws_ok && market_ok;
                    let detail = format!(
                        "{}:public_ws=true ws_url_ok={} market_ok={} api_key={} api_secret={}",
                        connector.as_str(),
                        ws_ok,
                        market_ok,
                        key_present,
                        secret_present
                    );
                    creds_detail.push_str(&detail);
                }
            }
        }
        creds_detail.push(' ');
    }
    creds_detail = creds_detail.trim_end().to_string();
    checks.push(PreflightCheck {
        label: "credentials",
        ok: creds_ok,
        details: creds_detail,
    });

    let live_guard_ok = if trade_mode == TradeMode::Live {
        connectors_allowed_for_live_mode(connectors)
            && args.enable_live_execution
            && env_is_true("PARAPHINA_LIVE_EXEC_ENABLE")
            && env_is_yes("PARAPHINA_LIVE_EXECUTION_CONFIRM")
    } else {
        true
    };
    checks.push(PreflightCheck {
        label: "live_guardrails",
        ok: live_guard_ok,
        details: if trade_mode == TradeMode::Live {
            format!(
                "enable_flag={} exec_env={} confirm_env={}",
                args.enable_live_execution,
                env_is_true("PARAPHINA_LIVE_EXEC_ENABLE"),
                env_is_yes("PARAPHINA_LIVE_EXECUTION_CONFIRM")
            )
        } else {
            "not_required".to_string()
        },
    });

    println!("paraphina_live preflight:");
    let mut failed = false;
    for check in &checks {
        let status = if check.ok { "PASS" } else { "FAIL" };
        println!("- {} {} {}", status, check.label, check.details);
        if !check.ok {
            failed = true;
        }
    }
    !failed
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    enforce_roadmap_b_gate();
    let effective = resolve_effective_profile(None, None);
    effective.log_startup();
    let mut cfg = Config::from_env_or_profile(effective.profile);
    let build_info = paraphina::BuildInfo::capture();
    let trade_mode = resolve_effective_trade_mode(args.trade_mode.map(TradeMode::from));
    trade_mode.log_startup();
    let tm_env: &str = match trade_mode.trade_mode {
        TradeMode::Shadow => "shadow",
        TradeMode::Paper => "paper",
        TradeMode::Testnet => "testnet",
        TradeMode::Live => "live",
    };
    std::env::set_var("PARAPHINA_TRADE_MODE", tm_env);
    let connector_selection = resolve_connectors(&args);
    let connectors = connector_selection.connectors.clone();
    let out_dir = resolve_out_dir(args.out_dir.clone());
    let paper_mode = trade_mode.trade_mode == TradeMode::Paper;
    let paper_route_sandbox = env_is_true("PARAPHINA_PAPER_ROUTE_SANDBOX");
    let canary_profile = resolve_canary_profile(args.canary_profile.clone());
    let mut canary_error: Option<String> = None;
    let mut canary_settings: Option<CanarySettings> = None;
    match resolve_canary_settings(
        trade_mode.trade_mode,
        &mut cfg,
        canary_profile.as_ref(),
        !args.preflight,
    ) {
        Ok(settings) => {
            canary_settings = settings;
        }
        Err(err) => {
            canary_error = Some(err.clone());
            if !args.preflight && trade_mode.trade_mode == TradeMode::Live {
                eprintln!("paraphina_live | error=canary_profile_load_failed {err}");
                std::process::exit(2);
            }
        }
    }
    if paper_mode {
        if let Ok(raw) = std::env::var("PARAPHINA_PAPER_MIN_HEALTHY_FOR_KF") {
            if let Ok(val) = raw.parse::<u32>() {
                let clamped = val.max(1);
                cfg.book.min_healthy_for_kf = clamped;
                eprintln!(
                    "paraphina_live | paper_mode_min_healthy_for_kf_override={}",
                    clamped
                );
            }
        }
    }
    if connector_selection.explicit_list {
        let selected: HashSet<&str> = connectors
            .iter()
            .map(|connector| connector_venue_id(*connector))
            .collect();
        cfg.venues
            .retain(|venue| selected.contains(venue.id.as_str()));
    }
    if args.preflight {
        let ok = run_preflight(
            &args,
            trade_mode.trade_mode,
            &connectors,
            &cfg,
            out_dir.clone(),
            canary_error.as_deref(),
            canary_settings.as_ref(),
        );
        std::process::exit(if ok { 0 } else { 1 });
    }
    enforce_live_execution_guardrails(
        &args,
        trade_mode.trade_mode,
        &connectors,
        canary_profile.as_ref(),
        canary_settings.as_ref(),
    );
    let metrics_addr = std::env::var("PARAPHINA_LIVE_METRICS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9898".to_string());
    let startup_log = format_startup_log(
        &cfg,
        &build_info,
        trade_mode.trade_mode,
        &connectors_label(&connectors),
        &metrics_addr,
    );
    println!("{startup_log}");
    eprintln!(
        "paraphina_live | trade_mode={} connectors={}",
        trade_mode.trade_mode.as_str(),
        connectors_label(&connectors)
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
    start_metrics_server(
        &metrics_addr,
        metrics.clone(),
        health.clone(),
        audit_dir.clone(),
    );

    let secrets = EnvSecretProvider::default();
    if secrets.get("PARAPHINA_LIVE_MODE").is_some() {
        // Secret provider is wired for future use.
    }

    let (market_ingest_tx, mut market_ingest_rx) =
        mpsc::channel::<paraphina::live::types::MarketDataEvent>(1024);
    let (market_tx, market_rx) = mpsc::channel::<paraphina::live::types::MarketDataEvent>(1024);
    let (paper_market_tx, paper_market_rx) = mpsc::channel::<PaperMarketUpdate>(1024);
    let paper_market_tx = if paper_mode {
        Some(paper_market_tx)
    } else {
        None
    };
    let override_market_ts = paper_mode && env_is_true("PARAPHINA_PAPER_USE_WALLCLOCK_TS");
    tokio::spawn(async move {
        while let Some(event) = market_ingest_rx.recv().await {
            let event = if override_market_ts {
                override_market_timestamp(event, now_ms())
            } else {
                event
            };
            if let Some(tx) = paper_market_tx.as_ref() {
                if let Some(update) = paper_market_update_from_event(&event) {
                    let _ = tx.send(update).await;
                }
            }
            let _ = market_tx.send(event).await;
        }
    });
    let (_account_tx, account_rx) = mpsc::channel::<paraphina::live::types::AccountEvent>(256);
    let (exec_tx, exec_rx) = mpsc::channel::<paraphina::live::types::ExecutionEvent>(512);
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

    let exec_enable_env = env_is_true("PARAPHINA_LIVE_EXEC_ENABLE");
    let allow_live_gateway = exec_enable_env
        && trade_mode.trade_mode != TradeMode::Shadow
        && (!paper_mode || paper_route_sandbox);
    let mut connector_channels: HashMap<ConnectorArg, ConnectorChannels> = HashMap::new();
    let mut connector_venues: HashMap<ConnectorArg, (String, usize)> = HashMap::new();
    for connector in &connectors {
        let (venue_id, venue_index) = match resolve_connector_venue(&cfg, *connector) {
            Ok((venue_id, venue_index)) => (venue_id, venue_index),
            Err(err) => {
                eprintln!("paraphina_live | error={err}");
                return;
            }
        };
        let (market_tx, market_rx) = mpsc::channel(1024);
        let (account_tx, account_rx) = mpsc::channel(256);
        let (exec_tx_local, exec_rx_local) = mpsc::channel(512);
        spawn_connector_forwarders(
            venue_id.clone(),
            venue_index,
            market_rx,
            account_rx,
            exec_rx_local,
            market_ingest_tx.clone(),
            _account_tx.clone(),
            exec_tx.clone(),
        );
        connector_channels.insert(
            *connector,
            ConnectorChannels {
                market_tx,
                account_tx,
                exec_tx: exec_tx_local,
            },
        );
        connector_venues.insert(*connector, (venue_id, venue_index));
    }

    let mut exec_clients: BTreeMap<String, Arc<dyn LiveRestClient>> = BTreeMap::new();
    for connector in &connectors {
        let support = connector_support(*connector);
        if matches!(support, ConnectorSupport::MissingFeature) {
            eprintln!(
                "paraphina_live | error=connector_unavailable connector={}",
                connector.as_str()
            );
            return;
        }
        if matches!(support, ConnectorSupport::Stub) {
            eprintln!(
                "paraphina_live | error=connector_stub connector={}",
                connector.as_str()
            );
            return;
        }
        let Some(channels) = connector_channels.get(connector) else {
            continue;
        };
        let (venue_id, venue_index) = connector_venues
            .get(connector)
            .cloned()
            .unwrap_or_else(|| (connector.as_str().to_string(), 0));
        match connector {
            ConnectorArg::Mock => {
                let market_tx_clone = channels.market_tx.clone();
                tokio::spawn(async move {
                    let mut seq: u64 = 0;
                    let mut mid = 100.0;
                    let mut interval = tokio::time::interval(Duration::from_millis(500));
                    loop {
                        interval.tick().await;
                        seq += 1;
                        mid += if seq % 2 == 0 { 0.1 } else { -0.1 };
                        let bids = vec![
                            BookLevel {
                                price: mid - 0.5,
                                size: 5.0,
                            },
                            BookLevel {
                                price: mid - 1.0,
                                size: 5.0,
                            },
                        ];
                        let asks = vec![
                            BookLevel {
                                price: mid + 0.5,
                                size: 5.0,
                            },
                            BookLevel {
                                price: mid + 1.0,
                                size: 5.0,
                            },
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
                            .send(paraphina::live::types::MarketDataEvent::L2Snapshot(
                                snapshot,
                            ))
                            .await;
                    }
                });
            }
            ConnectorArg::Hyperliquid => {
                #[cfg(feature = "live_hyperliquid")]
                {
                    let mut hl_cfg =
                        paraphina::live::connectors::hyperliquid::HyperliquidConfig::from_env();
                    hl_cfg.venue_index = venue_index;
                    let mut hl =
                        paraphina::live::connectors::hyperliquid::HyperliquidConnector::new(
                            hl_cfg.clone(),
                            channels.market_tx.clone(),
                            channels.exec_tx.clone(),
                        );
                    if trade_mode.trade_mode != TradeMode::Shadow {
                        if hl_cfg.vault_address.is_some() {
                            let account_tx = channels.account_tx.clone();
                            hl = hl.with_account_tx(account_tx);
                        } else {
                            eprintln!(
                                "paraphina_live | account_snapshots_disabled=true reason=missing_hl_vault_address connector=hyperliquid"
                            );
                            if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                            }
                        }
                    } else {
                        eprintln!(
                            "paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=hyperliquid"
                        );
                        if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                            send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                        }
                    }
                    let hl_arc = Arc::new(hl);
                    if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                        if hl_cfg.private_key_hex.is_some() {
                            exec_clients.insert(venue_id.clone(), hl_arc.clone());
                        } else {
                            eprintln!("paraphina_live | exec_disabled=true reason=missing_hl_private_key connector=hyperliquid");
                        }
                    }
                    if trade_mode.trade_mode != TradeMode::Shadow && hl_cfg.vault_address.is_some()
                    {
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
                            eprintln!("paraphina_live | private_ws_disabled=true reason=missing_hl_private_key connector=hyperliquid");
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
                        .unwrap_or_else(|_| {
                            std::path::PathBuf::from("./tests/fixtures/hyperliquid")
                        });
                    match paraphina::live::connectors::hyperliquid::HyperliquidFixtureFeed::from_dir(
                        &fixture_dir,
                    ) {
                        Ok(feed) => {
                            let market_tx = channels.market_tx.clone();
                            tokio::spawn(async move {
                                feed.run_ticks(market_tx, venue_index, 1_000, 250, 200)
                                    .await;
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
                        let account_tx = channels.account_tx.clone();
                        tokio::spawn(async move {
                            feed.run_ticks(account_tx, 1_000, 250, 200).await;
                        });
                    }
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
                    let mut lighter_cfg =
                        paraphina::live::connectors::lighter::LighterConfig::from_env();
                    lighter_cfg.venue_index = venue_index;
                    let mut lighter = paraphina::live::connectors::lighter::LighterConnector::new(
                        lighter_cfg.clone(),
                        channels.market_tx.clone(),
                        channels.exec_tx.clone(),
                    );
                    if trade_mode.trade_mode != TradeMode::Shadow {
                        if lighter_cfg.paper_mode {
                            eprintln!("paraphina_live | account_snapshots_disabled=true reason=lighter_paper_mode connector=lighter");
                            if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                            }
                        } else {
                            lighter = lighter.with_account_tx(channels.account_tx.clone());
                        }
                    } else {
                        eprintln!("paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=lighter");
                        if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                            send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                        }
                    }
                    let lighter_arc = Arc::new(lighter);
                    let fixture_dir = std::env::var("LIGHTER_FIXTURE_DIR")
                        .ok()
                        .map(std::path::PathBuf::from);
                    if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                        if lighter_cfg.paper_mode {
                            eprintln!("paraphina_live | exec_disabled=true reason=lighter_paper_mode connector=lighter");
                        } else {
                            exec_clients.insert(venue_id.clone(), lighter_arc.clone());
                        }
                    }
                    if let Some(fixture_dir) = fixture_dir {
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            let fixture_dir_clone = fixture_dir.clone();
                            let lighter_fixture = lighter_arc.clone();
                            tokio::spawn(async move {
                                lighter_fixture
                                    .run_account_fixture(&fixture_dir_clone, 1_000, 250, 200)
                                    .await;
                            });
                        }
                        if let Ok(feed) =
                            paraphina::live::connectors::lighter::LighterFixtureFeed::from_dir(
                                &fixture_dir,
                            )
                        {
                            let market_tx = channels.market_tx.clone();
                            tokio::spawn(async move {
                                feed.run_ticks(market_tx, venue_index, 1_000, 250, 200)
                                    .await;
                            });
                        } else {
                            eprintln!(
                                "paraphina_live | warn=lighter_fixture_missing dir={}",
                                fixture_dir.display()
                            );
                        }
                    } else {
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
                }
                #[cfg(not(feature = "live_lighter"))]
                {
                    eprintln!("paraphina_live | error=connector_unavailable connector=lighter");
                    return;
                }
            }
            ConnectorArg::Extended => {
                #[cfg(feature = "live_extended")]
                {
                    if extended_fixture_mode(&args) {
                        let Some(fixture_dir) = resolve_fixture_dir(*connector) else {
                            eprintln!(
                                "paraphina_live | error=fixture_dir_missing connector=extended"
                            );
                            return;
                        };
                        match paraphina::live::connectors::extended::ExtendedFixtureFeed::from_dir(
                            &fixture_dir,
                        ) {
                            Ok(feed) => {
                                let market_tx = channels.market_tx.clone();
                                let account_tx = channels.account_tx.clone();
                                let venue_id = venue_id.clone();
                                tokio::spawn(async move {
                                    feed.run_ticks(
                                        market_tx,
                                        account_tx,
                                        &venue_id,
                                        venue_index,
                                        1_000,
                                        250,
                                        200,
                                    )
                                    .await;
                                });
                            }
                            Err(err) => {
                                eprintln!(
                                    "paraphina_live | error=fixture_dir_unreadable connector=extended dir={} err={}",
                                    fixture_dir.display(),
                                    err
                                );
                                return;
                            }
                        }
                    } else {
                        let mut extended_cfg =
                            paraphina::live::connectors::extended::ExtendedConfig::from_env();
                        extended_cfg.venue_index = venue_index;
                        if extended_record_enabled(&args) {
                            extended_cfg =
                                extended_cfg.with_record_dir(resolve_extended_record_dir());
                        }
                        let rest_client = Arc::new(
                            paraphina::live::connectors::extended::ExtendedRestClient::new(
                                extended_cfg.clone(),
                            ),
                        );
                        let extended =
                            paraphina::live::connectors::extended::ExtendedConnector::new(
                                extended_cfg,
                                channels.market_tx.clone(),
                            );
                        let extended_arc = Arc::new(extended);
                        let extended_public = extended_arc.clone();
                        tokio::spawn(async move {
                            extended_public.run_public_ws().await;
                        });
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                                    .ok()
                                    .and_then(|v| v.parse::<u64>().ok())
                                    .unwrap_or(5_000);
                                let account_tx = channels.account_tx.clone();
                                let venue_id = venue_id.clone();
                                let rest_client_clone = rest_client.clone();
                                tokio::spawn(async move {
                                    rest_client_clone
                                        .run_account_polling(
                                            account_tx,
                                            venue_id,
                                            venue_index,
                                            poll_ms,
                                        )
                                        .await;
                                });
                            } else {
                                eprintln!(
                                    "paraphina_live | account_snapshots_disabled=true reason=missing_extended_api_keys connector=extended"
                                );
                                if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                    send_unavailable_account_snapshot_for(
                                        &_account_tx,
                                        &cfg,
                                        index,
                                    );
                                }
                            }
                        }
                        if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                exec_clients.insert(venue_id.clone(), rest_client.clone());
                            } else {
                                eprintln!(
                                    "paraphina_live | exec_disabled=true reason=missing_extended_api_keys connector=extended"
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "live_extended"))]
                {
                    eprintln!("paraphina_live | error=connector_unavailable connector=extended feature=live_extended");
                    return;
                }
            }
            ConnectorArg::Aster => {
                #[cfg(feature = "live_aster")]
                {
                    if aster_fixture_mode(&args) {
                        let Some(fixture_dir) = resolve_fixture_dir(*connector) else {
                            eprintln!("paraphina_live | error=fixture_dir_missing connector=aster");
                            return;
                        };
                        match paraphina::live::connectors::aster::AsterFixtureFeed::from_dir(
                            &fixture_dir,
                        ) {
                            Ok(feed) => {
                                let market_tx = channels.market_tx.clone();
                                let account_tx = channels.account_tx.clone();
                                let venue_id = venue_id.clone();
                                tokio::spawn(async move {
                                    feed.run_ticks(
                                        market_tx,
                                        account_tx,
                                        &venue_id,
                                        venue_index,
                                        1_000,
                                        250,
                                        200,
                                    )
                                    .await;
                                });
                            }
                            Err(err) => {
                                eprintln!(
                                    "paraphina_live | error=fixture_dir_unreadable connector=aster dir={} err={}",
                                    fixture_dir.display(),
                                    err
                                );
                                return;
                            }
                        }
                    } else {
                        let mut aster_cfg =
                            paraphina::live::connectors::aster::AsterConfig::from_env();
                        aster_cfg.venue_index = venue_index;
                        if aster_record_enabled(&args) {
                            aster_cfg = aster_cfg.with_record_dir(resolve_aster_record_dir());
                        }
                        let rest_client =
                            Arc::new(paraphina::live::connectors::aster::AsterRestClient::new(
                                aster_cfg.clone(),
                            ));
                        let aster = paraphina::live::connectors::aster::AsterConnector::new(
                            aster_cfg,
                            channels.market_tx.clone(),
                        );
                        let aster_arc = Arc::new(aster);
                        let aster_public = aster_arc.clone();
                        tokio::spawn(async move {
                            aster_public.run_public_ws().await;
                        });
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                                    .ok()
                                    .and_then(|v| v.parse::<u64>().ok())
                                    .unwrap_or(5_000);
                                let account_tx = channels.account_tx.clone();
                                let venue_id = venue_id.clone();
                                let rest_client_clone = rest_client.clone();
                                tokio::spawn(async move {
                                    rest_client_clone
                                        .run_account_polling(
                                            account_tx,
                                            venue_id,
                                            venue_index,
                                            poll_ms,
                                        )
                                        .await;
                                });
                            } else {
                                eprintln!(
                                    "paraphina_live | account_snapshots_disabled=true reason=missing_aster_api_keys connector=aster"
                                );
                                if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                    send_unavailable_account_snapshot_for(
                                        &_account_tx,
                                        &cfg,
                                        index,
                                    );
                                }
                            }
                        }
                        if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                exec_clients.insert(venue_id.clone(), rest_client.clone());
                            } else {
                                eprintln!(
                                    "paraphina_live | exec_disabled=true reason=missing_aster_api_keys connector=aster"
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "live_aster"))]
                {
                    eprintln!("paraphina_live | error=connector_unavailable connector=aster feature=live_aster");
                    return;
                }
            }
            ConnectorArg::Paradex => {
                #[cfg(feature = "live_paradex")]
                {
                    if paradex_fixture_mode(&args) {
                        let Some(fixture_dir) = resolve_fixture_dir(*connector) else {
                            eprintln!(
                                "paraphina_live | error=fixture_dir_missing connector=paradex"
                            );
                            return;
                        };
                        match paraphina::live::connectors::paradex::ParadexFixtureFeed::from_dir(
                            &fixture_dir,
                        ) {
                            Ok(feed) => {
                                let market_tx = channels.market_tx.clone();
                                let account_tx = channels.account_tx.clone();
                                let venue_id = venue_id.clone();
                                tokio::spawn(async move {
                                    feed.run_ticks(
                                        market_tx,
                                        account_tx,
                                        &venue_id,
                                        venue_index,
                                        1_000,
                                        250,
                                        200,
                                    )
                                    .await;
                                });
                            }
                            Err(err) => {
                                eprintln!(
                                    "paraphina_live | error=fixture_dir_unreadable connector=paradex dir={} err={}",
                                    fixture_dir.display(),
                                    err
                                );
                                return;
                            }
                        }
                    } else {
                        let mut paradex_cfg =
                            paraphina::live::connectors::paradex::ParadexConfig::from_env();
                        paradex_cfg.venue_index = venue_index;
                        if paradex_record_enabled(&args) {
                            paradex_cfg = paradex_cfg.with_record_dir(resolve_paradex_record_dir());
                        }
                        let rest_client = Arc::new(
                            paraphina::live::connectors::paradex::ParadexRestClient::new(
                                paradex_cfg.clone(),
                            ),
                        );
                        let paradex = paraphina::live::connectors::paradex::ParadexConnector::new(
                            paradex_cfg,
                            channels.market_tx.clone(),
                        );
                        let paradex_arc = Arc::new(paradex);
                        let paradex_public = paradex_arc.clone();
                        tokio::spawn(async move {
                            paradex_public.run_public_ws().await;
                        });
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                                    .ok()
                                    .and_then(|v| v.parse::<u64>().ok())
                                    .unwrap_or(5_000);
                                let account_tx = channels.account_tx.clone();
                                let venue_id = venue_id.clone();
                                let rest_client_clone = rest_client.clone();
                                tokio::spawn(async move {
                                    rest_client_clone
                                        .run_account_polling(
                                            account_tx,
                                            venue_id,
                                            venue_index,
                                            poll_ms,
                                        )
                                        .await;
                                });
                            } else {
                                eprintln!(
                                    "paraphina_live | account_snapshots_disabled=true reason=missing_paradex_auth connector=paradex"
                                );
                                if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                    send_unavailable_account_snapshot_for(
                                        &_account_tx,
                                        &cfg,
                                        index,
                                    );
                                }
                            }
                        }
                        if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                exec_clients.insert(venue_id.clone(), rest_client.clone());
                            } else {
                                eprintln!(
                                    "paraphina_live | exec_disabled=true reason=missing_paradex_auth connector=paradex"
                                );
                            }
                        }
                    }
                }
                #[cfg(not(feature = "live_paradex"))]
                {
                    eprintln!("paraphina_live | error=connector_unavailable connector=paradex feature=live_paradex");
                    return;
                }
            }
        }
    }

    let exec_enabled = allow_live_gateway && !exec_clients.is_empty();
    if trade_mode.trade_mode != TradeMode::Shadow && !exec_enabled {
        if paper_mode && !paper_route_sandbox {
            eprintln!(
                "paraphina_live | trade_mode=paper | paper_execution=internal | exec_disabled=true"
            );
        } else {
            eprintln!(
                "paraphina_live | trade_mode={} | exec_disabled=true | falling_back=shadow (set PARAPHINA_LIVE_EXEC_ENABLE=1 and provide keys)",
                trade_mode.trade_mode.as_str()
            );
        }
    }

    let exec_trade_mode = trade_mode.trade_mode;
    let exec_cfg = cfg.clone();
    let venue_id_lookup: Vec<String> = cfg.venues.iter().map(|v| v.id.clone()).collect();
    let exec_enabled_flag = exec_enabled;
    let exec_client: Option<Arc<dyn LiveRestClient>> = if exec_enabled_flag {
        Some(Arc::new(GatewayMux::new(exec_clients)))
    } else {
        None
    };
    let exec_metrics = metrics.clone();
    let exec_tx = exec_tx.clone();
    let account_tx = _account_tx.clone();
    let use_paper_adapter = paper_mode && !exec_enabled_flag;
    tokio::spawn(async move {
        let mut shadow = ShadowAckAdapter::new(&exec_cfg);
        let mut paper_adapter = if use_paper_adapter {
            Some(PaperExecutionAdapter::new(&exec_cfg))
        } else {
            None
        };
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
                    eprintln!(
                        "paraphina_live | exec_gateway_error={} fallback=shadow",
                        err.message
                    );
                    None
                }
            }
        } else {
            None
        };
        if let Some(adapter) = paper_adapter.as_ref() {
            let mode_label = match adapter.config().fill_mode {
                PaperFillMode::None => "none",
                PaperFillMode::Marketable => "marketable",
                PaperFillMode::Mid => "mid",
                PaperFillMode::Always => "always",
            };
            eprintln!(
                "paraphina_live | paper_execution=internal fill_mode={} slippage_bps={}",
                mode_label,
                adapter.config().slippage_bps
            );
        }
        let mut exec_seq: u64 = 0;
        let mut paper_market_rx = if use_paper_adapter {
            Some(paper_market_rx)
        } else {
            None
        };
        loop {
            tokio::select! {
                Some(update) = async {
                    if let Some(rx) = paper_market_rx.as_mut() {
                        rx.recv().await
                    } else {
                        None
                    }
                } => {
                    if let Some(adapter) = paper_adapter.as_mut() {
                        let events = adapter.update_best_bid_ask(update);
                        for event in events {
                            let _ = exec_tx.try_send(event);
                        }
                    }
                }
                Some(req) = order_rx.recv() => {
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
                    } else if let Some(adapter) = paper_adapter.as_mut() {
                        let events = adapter.handle_intents(intents, action_batch.tick_index, now_ms);
                        let mut response_events = Vec::new();
                        for event in events {
                            match &event {
                                paraphina::live::types::ExecutionEvent::Filled(_) => {
                                    let _ = exec_tx.try_send(event);
                                }
                                _ => response_events.push(event),
                            }
                        }
                        let snapshots = adapter.drain_account_snapshots(&venue_id_lookup, now_ms + 1);
                        for snapshot in snapshots {
                            let _ = account_tx.try_send(paraphina::live::types::AccountEvent::Snapshot(snapshot));
                        }
                        response_events
                    } else {
                        shadow.handle_intents(intents, action_batch.tick_index, now_ms)
                    };
                    let _ = response.send(events);
                }
                else => break,
            }
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
    let telemetry_path = resolve_telemetry_path(out_dir.as_ref());
    let telemetry_cfg = TelemetryConfig {
        mode: TelemetryMode::from_env(),
        path: telemetry_path,
        append: TelemetryConfig::append_from_env(),
    };
    let telemetry = LiveTelemetry {
        sink: Arc::new(Mutex::new(TelemetrySink::from_config(telemetry_cfg))),
        shadow_mode: trade_mode.trade_mode == TradeMode::Shadow,
        execution_mode: trade_mode.trade_mode.as_str(),
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
            &connectors_label(&connectors),
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
        .submit_intent(
            &paraphina::types::OrderIntent::Place(place.clone()),
            tick,
            now_ms,
        )
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
    let res = gateway
        .submit_intent(
            &paraphina::types::OrderIntent::Cancel(cancel.clone()),
            tick,
            now_ms,
        )
        .await;
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

fn send_unavailable_account_snapshot_for(
    account_tx: &mpsc::Sender<paraphina::live::types::AccountEvent>,
    cfg: &Config,
    venue_index: usize,
) {
    let Some(venue) = cfg.venues.get(venue_index) else {
        return;
    };
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
        "execution_mode": trade_mode.as_str(),
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
    use super::{connector_support, ConnectorArg, ConnectorSupport};
    use paraphina::live::venues::ROADMAP_B_VENUES;

    #[test]
    fn roadmap_b_registry_is_complete() {
        assert_eq!(ROADMAP_B_VENUES.len(), 5);
        let selectable = ConnectorArg::roadmap_b_selectable_venues();
        assert_eq!(selectable, ROADMAP_B_VENUES.to_vec());
    }

    #[test]
    fn roadmap_b_cli_selection_recognizes_all() {
        let connectors = [
            ConnectorArg::Extended,
            ConnectorArg::Hyperliquid,
            ConnectorArg::Aster,
            ConnectorArg::Lighter,
            ConnectorArg::Paradex,
        ];
        let parsed: Vec<&str> = connectors
            .iter()
            .map(|connector| ConnectorArg::parse_env(connector.as_str()).expect("parse env"))
            .map(|connector| connector.roadmap_b_venue_id().expect("venue id"))
            .collect();
        assert_eq!(parsed, ROADMAP_B_VENUES.to_vec());
    }

    #[cfg(feature = "live_hyperliquid")]
    #[test]
    fn roadmap_b_feature_live_hyperliquid_enabled() {
        assert!(cfg!(feature = "live_hyperliquid"));
        assert_eq!(
            connector_support(ConnectorArg::Hyperliquid),
            ConnectorSupport::MarketAccountExec
        );
    }

    #[cfg(feature = "live_lighter")]
    #[test]
    fn roadmap_b_feature_live_lighter_enabled() {
        assert!(cfg!(feature = "live_lighter"));
        assert_eq!(
            connector_support(ConnectorArg::Lighter),
            ConnectorSupport::MarketAccountExec
        );
    }

    #[cfg(feature = "live_extended")]
    #[test]
    fn roadmap_b_feature_live_extended_enabled() {
        assert!(cfg!(feature = "live_extended"));
        assert_eq!(
            connector_support(ConnectorArg::Extended),
            ConnectorSupport::MarketAccountExec
        );
    }

    #[cfg(feature = "live_aster")]
    #[test]
    fn roadmap_b_feature_live_aster_enabled() {
        assert!(cfg!(feature = "live_aster"));
        assert_eq!(
            connector_support(ConnectorArg::Aster),
            ConnectorSupport::MarketAccountExec
        );
    }

    #[cfg(feature = "live_paradex")]
    #[test]
    fn roadmap_b_feature_live_paradex_enabled() {
        assert!(cfg!(feature = "live_paradex"));
        assert_eq!(
            connector_support(ConnectorArg::Paradex),
            ConnectorSupport::MarketAccountExec
        );
    }
}
