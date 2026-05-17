//! Live trading skeleton binary (feature-gated).
//!
//! This binary wires the live cache, event model, and strategy loop together
//! without any external network connectors.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use paraphina::config::{resolve_effective_profile, Config};
use paraphina::io::GatewayPolicy;
use paraphina::live::gateway::{GatewayMux, LiveGateway, LiveRestClient, TransportHint};
use paraphina::live::instrument::{validate_specs, InstrumentSpec};
use paraphina::live::ops::{
    default_audit_dir, format_startup_log, start_metrics_server, write_audit_files,
    EnvSecretProvider, HealthState, LiveMetrics, SecretProvider,
};
use paraphina::live::orderbook_l2::BookLevel;
use paraphina::live::paper_adapter::{PaperExecutionAdapter, PaperFillMode, PaperMarketUpdate};
use paraphina::live::runner::{
    run_live_loop, LiveAccountRequest, LiveChannels, LiveOrderRequest, LiveRunMode,
    LiveRuntimeHooks, ResponseMode,
};
use paraphina::live::shadow_adapter::ShadowAckAdapter;
use paraphina::live::supervision::spawn_supervised;
use paraphina::live::types::L2Snapshot;
use paraphina::live::venues::{canonical_venue_ids, roadmap_b_enabled};
use paraphina::live::{resolve_effective_trade_mode, LiveTelemetry, LiveTelemetryStats, TradeMode};
use paraphina::telemetry::{TelemetryConfig, TelemetryMode, TelemetrySink, TelemetrySinkHandle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use url::Url;

use tokio::sync::mpsc;

type AccountRefreshFetch = Arc<
    dyn Fn() -> paraphina::live::gateway::BoxFuture<
            'static,
            paraphina::live::gateway::LiveResult<paraphina::live::types::AccountSnapshot>,
        > + Send
        + Sync,
>;

#[derive(Clone)]
struct AccountRefreshHandler {
    connector: &'static str,
    venue_id: String,
    venue_index: usize,
    fetch: AccountRefreshFetch,
}

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
    /// Validate config by loading it, constructing Engine + GlobalState,
    /// running one synthetic tick, and exiting. Exit 0 on success, 1 on failure.
    #[arg(long)]
    validate_config: bool,
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

fn apply_explicit_connector_selection_to_config(cfg: &mut Config, connectors: &[ConnectorArg]) {
    let selected: HashSet<&str> = connectors
        .iter()
        .map(|connector| connector_venue_id(*connector))
        .collect();
    cfg.venues
        .retain(|venue| selected.contains(venue.id.as_str()));

    let selected_venue_count = cfg.venues.len() as u32;
    if selected_venue_count > 0 && cfg.book.min_healthy_for_kf > selected_venue_count {
        cfg.book.min_healthy_for_kf = selected_venue_count;
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
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
    soft_max_position_tao: Option<f64>,
    soft_max_gross_position_tao: Option<f64>,
    soft_max_abs_venue_position_tao: Option<f64>,
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
struct CanaryBookConfig {
    max_mid_jump_pct: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CanaryConfig {
    venue: Option<CanaryVenueConfig>,
    limits: Option<CanaryLimitsConfig>,
    rate_limit: Option<CanaryRateLimitConfig>,
    enforcement: Option<CanaryEnforcementConfig>,
    kill: Option<CanaryKillConfig>,
    book: Option<CanaryBookConfig>,
}

#[derive(Debug, Default, Clone)]
struct CanarySettings {
    max_position_tao: Option<f64>,
    max_gross_position_tao: Option<f64>,
    max_abs_venue_position_tao: Option<f64>,
    soft_max_position_tao: Option<f64>,
    soft_max_gross_position_tao: Option<f64>,
    soft_max_abs_venue_position_tao: Option<f64>,
    max_open_orders: Option<usize>,
    stale_max_ticks: Option<u64>,
    enforce_post_only: bool,
    enforce_reduce_only: bool,
    rate_limit_enabled: Option<bool>,
    rate_limit_rps: Option<f64>,
    rate_limit_burst: Option<u32>,
    max_mid_jump_pct: Option<f64>,
}

const RUNTIME_CANARY_PROFILE_PATH_ENV: &str = "PARAPHINA_RUNTIME_CANARY_PROFILE_PATH";
const RUNTIME_CANARY_PROFILE_SHA256_ENV: &str = "PARAPHINA_RUNTIME_CANARY_PROFILE_SHA256";
const RUNTIME_CANARY_MAX_POSITION_ENV: &str = "PARAPHINA_RUNTIME_CANARY_MAX_POSITION_TAO";
const RUNTIME_CANARY_MAX_GROSS_POSITION_ENV: &str =
    "PARAPHINA_RUNTIME_CANARY_MAX_GROSS_POSITION_TAO";
const RUNTIME_CANARY_MAX_ABS_VENUE_POSITION_ENV: &str =
    "PARAPHINA_RUNTIME_CANARY_MAX_ABS_VENUE_POSITION_TAO";
const RUNTIME_CANARY_SOFT_MAX_POSITION_ENV: &str = "PARAPHINA_RUNTIME_CANARY_SOFT_MAX_POSITION_TAO";
const RUNTIME_CANARY_SOFT_MAX_GROSS_POSITION_ENV: &str =
    "PARAPHINA_RUNTIME_CANARY_SOFT_MAX_GROSS_POSITION_TAO";
const RUNTIME_CANARY_SOFT_MAX_ABS_VENUE_POSITION_ENV: &str =
    "PARAPHINA_RUNTIME_CANARY_SOFT_MAX_ABS_VENUE_POSITION_TAO";
const RUNTIME_CANARY_MAX_OPEN_ORDERS_ENV: &str = "PARAPHINA_RUNTIME_CANARY_MAX_OPEN_ORDERS";
const RUNTIME_CANARY_STALE_MAX_TICKS_ENV: &str = "PARAPHINA_RUNTIME_CANARY_STALE_MAX_TICKS";
const RUNTIME_CANARY_POST_ONLY_ENV: &str = "PARAPHINA_RUNTIME_CANARY_ENFORCE_POST_ONLY";
const RUNTIME_CANARY_REDUCE_ONLY_ENV: &str = "PARAPHINA_RUNTIME_CANARY_ENFORCE_REDUCE_ONLY";
const RUNTIME_CANARY_RATE_LIMIT_ENABLED_ENV: &str = "PARAPHINA_RUNTIME_CANARY_RATE_LIMIT_ENABLED";
const RUNTIME_CANARY_RATE_LIMIT_RPS_ENV: &str = "PARAPHINA_RUNTIME_CANARY_RATE_LIMIT_RPS";
const RUNTIME_CANARY_RATE_LIMIT_BURST_ENV: &str = "PARAPHINA_RUNTIME_CANARY_RATE_LIMIT_BURST";

fn hash_file_sha256(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Some(format!("{:x}", hasher.finalize()))
}

fn clear_runtime_canary_metadata() {
    for key in [
        RUNTIME_CANARY_PROFILE_PATH_ENV,
        RUNTIME_CANARY_PROFILE_SHA256_ENV,
        RUNTIME_CANARY_MAX_POSITION_ENV,
        RUNTIME_CANARY_MAX_GROSS_POSITION_ENV,
        RUNTIME_CANARY_MAX_ABS_VENUE_POSITION_ENV,
        RUNTIME_CANARY_SOFT_MAX_POSITION_ENV,
        RUNTIME_CANARY_SOFT_MAX_GROSS_POSITION_ENV,
        RUNTIME_CANARY_SOFT_MAX_ABS_VENUE_POSITION_ENV,
        RUNTIME_CANARY_MAX_OPEN_ORDERS_ENV,
        RUNTIME_CANARY_STALE_MAX_TICKS_ENV,
        RUNTIME_CANARY_POST_ONLY_ENV,
        RUNTIME_CANARY_REDUCE_ONLY_ENV,
        RUNTIME_CANARY_RATE_LIMIT_ENABLED_ENV,
        RUNTIME_CANARY_RATE_LIMIT_RPS_ENV,
        RUNTIME_CANARY_RATE_LIMIT_BURST_ENV,
    ] {
        std::env::remove_var(key);
    }
}

fn export_runtime_canary_metadata(
    canary_profile: Option<&PathBuf>,
    canary_settings: Option<&CanarySettings>,
) {
    clear_runtime_canary_metadata();
    let resolved_settings = resolved_canary_settings_for_metadata(canary_profile, canary_settings);

    if let Some(path) = canary_profile {
        std::env::set_var(RUNTIME_CANARY_PROFILE_PATH_ENV, path.display().to_string());
        if let Some(sha) = hash_file_sha256(path) {
            std::env::set_var(RUNTIME_CANARY_PROFILE_SHA256_ENV, sha);
        }
    }

    if let Some(settings) = resolved_settings.as_ref() {
        if let Some(val) = settings.max_position_tao {
            std::env::set_var(RUNTIME_CANARY_MAX_POSITION_ENV, val.to_string());
        }
        if let Some(val) = settings.max_gross_position_tao {
            std::env::set_var(RUNTIME_CANARY_MAX_GROSS_POSITION_ENV, val.to_string());
        }
        if let Some(val) = settings.max_abs_venue_position_tao {
            std::env::set_var(RUNTIME_CANARY_MAX_ABS_VENUE_POSITION_ENV, val.to_string());
        }
        if let Some(val) = settings.soft_max_position_tao {
            std::env::set_var(RUNTIME_CANARY_SOFT_MAX_POSITION_ENV, val.to_string());
        }
        if let Some(val) = settings.soft_max_gross_position_tao {
            std::env::set_var(RUNTIME_CANARY_SOFT_MAX_GROSS_POSITION_ENV, val.to_string());
        }
        if let Some(val) = settings.soft_max_abs_venue_position_tao {
            std::env::set_var(
                RUNTIME_CANARY_SOFT_MAX_ABS_VENUE_POSITION_ENV,
                val.to_string(),
            );
        }
        if let Some(val) = settings.max_open_orders {
            std::env::set_var(RUNTIME_CANARY_MAX_OPEN_ORDERS_ENV, val.to_string());
        }
        if let Some(val) = settings.stale_max_ticks {
            std::env::set_var(RUNTIME_CANARY_STALE_MAX_TICKS_ENV, val.to_string());
        }
        std::env::set_var(
            RUNTIME_CANARY_POST_ONLY_ENV,
            if settings.enforce_post_only { "1" } else { "0" },
        );
        std::env::set_var(
            RUNTIME_CANARY_REDUCE_ONLY_ENV,
            if settings.enforce_reduce_only {
                "1"
            } else {
                "0"
            },
        );
        if let Some(val) = settings.rate_limit_enabled {
            std::env::set_var(
                RUNTIME_CANARY_RATE_LIMIT_ENABLED_ENV,
                if val { "1" } else { "0" },
            );
        }
        if let Some(val) = settings.rate_limit_rps {
            std::env::set_var(RUNTIME_CANARY_RATE_LIMIT_RPS_ENV, val.to_string());
        }
        if let Some(val) = settings.rate_limit_burst {
            std::env::set_var(RUNTIME_CANARY_RATE_LIMIT_BURST_ENV, val.to_string());
        }
    }
}

fn format_runtime_canary_log(
    canary_profile: Option<&PathBuf>,
    canary_settings: Option<&CanarySettings>,
) -> Option<String> {
    let profile = canary_profile?;
    let sha = hash_file_sha256(profile).unwrap_or_else(|| "unknown".to_string());
    let settings = resolved_canary_settings_for_metadata(canary_profile, canary_settings)
        .as_ref()
        .map(|settings| {
            format!(
                " max_position_tao={} max_gross_position_tao={} max_abs_venue_position_tao={} soft_max_position_tao={} soft_max_gross_position_tao={} soft_max_abs_venue_position_tao={} max_open_orders={} stale_max_ticks={} post_only={} reduce_only={} rate_limit_rps={} rate_limit_burst={} max_mid_jump_pct={}",
                settings
                    .max_position_tao
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .max_gross_position_tao
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .max_abs_venue_position_tao
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .soft_max_position_tao
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .soft_max_gross_position_tao
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .soft_max_abs_venue_position_tao
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .max_open_orders
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .stale_max_ticks
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings.enforce_post_only,
                settings.enforce_reduce_only,
                settings
                    .rate_limit_rps
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .rate_limit_burst
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
                settings
                    .max_mid_jump_pct
                    .map(|val| val.to_string())
                    .unwrap_or_else(|| "none".to_string()),
            )
        })
        .unwrap_or_default();
    Some(format!(
        "paraphina_live | canary_profile={} | canary_sha256={}{}",
        profile.display(),
        sha,
        settings
    ))
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

fn canary_settings_from_config(canary: &CanaryConfig) -> CanarySettings {
    let mut settings = CanarySettings::default();
    if let Some(limits) = &canary.limits {
        settings.max_position_tao = limits.max_position_tao;
        settings.max_gross_position_tao = limits.max_gross_position_tao;
        settings.max_abs_venue_position_tao = limits.max_abs_venue_position_tao;
        settings.soft_max_position_tao = limits.soft_max_position_tao;
        settings.soft_max_gross_position_tao = limits.soft_max_gross_position_tao;
        settings.soft_max_abs_venue_position_tao = limits.soft_max_abs_venue_position_tao;
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
    if let Some(book) = &canary.book {
        settings.max_mid_jump_pct = book.max_mid_jump_pct;
    }
    settings
}

fn apply_canary_config(cfg: &mut Config, canary: &CanaryConfig) -> CanarySettings {
    let settings = canary_settings_from_config(canary);
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
    if let Some(max_mid_jump_pct) = settings.max_mid_jump_pct {
        if max_mid_jump_pct.is_finite() && max_mid_jump_pct > 0.0 {
            cfg.book.max_mid_jump_pct = max_mid_jump_pct;
        }
    }
    settings
}

fn resolved_canary_settings_for_metadata(
    canary_profile: Option<&PathBuf>,
    canary_settings: Option<&CanarySettings>,
) -> Option<CanarySettings> {
    canary_settings.cloned().or_else(|| {
        let path = canary_profile?;
        let canary = load_canary_config(path).ok()?;
        Some(canary_settings_from_config(&canary))
    })
}

fn apply_canary_env(settings: &CanarySettings) {
    std::env::set_var("PARAPHINA_CANARY_MODE", "1");
    if let Some(val) = settings.max_position_tao {
        std::env::set_var("PARAPHINA_CANARY_MAX_POSITION_TAO", val.to_string());
    }
    if let Some(val) = settings.max_gross_position_tao {
        std::env::set_var("PARAPHINA_CANARY_MAX_GROSS_POSITION_TAO", val.to_string());
    }
    if let Some(val) = settings.max_abs_venue_position_tao {
        std::env::set_var(
            "PARAPHINA_CANARY_MAX_ABS_VENUE_POSITION_TAO",
            val.to_string(),
        );
    }
    if let Some(val) = settings.soft_max_position_tao {
        std::env::set_var("PARAPHINA_CANARY_SOFT_MAX_POSITION_TAO", val.to_string());
    }
    if let Some(val) = settings.soft_max_gross_position_tao {
        std::env::set_var(
            "PARAPHINA_CANARY_SOFT_MAX_GROSS_POSITION_TAO",
            val.to_string(),
        );
    }
    if let Some(val) = settings.soft_max_abs_venue_position_tao {
        std::env::set_var(
            "PARAPHINA_CANARY_SOFT_MAX_ABS_VENUE_POSITION_TAO",
            val.to_string(),
        );
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

fn should_fail_on_unexpected_live_loop_exit(mode: LiveRunMode) -> bool {
    matches!(
        mode,
        LiveRunMode::Realtime {
            max_ticks: None,
            ..
        }
    )
}

const CURRENT_RUN_POINTER_PATH: &str = "/tmp/paraphina_current_run.json";
const CURRENT_RUNS_DIR: &str = "/tmp/paraphina_current_runs";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CurrentRunManifest {
    pid: u32,
    started_at: String,
    started_at_unix_ms: u64,
    trade_mode: String,
    telemetry_path: String,
    out_dir: Option<String>,
    manifest_path: String,
}

#[derive(Debug)]
struct CurrentRunRegistration {
    pid: u32,
    manifest_path: PathBuf,
    pointer_path: PathBuf,
}

impl Drop for CurrentRunRegistration {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.manifest_path);
        let current = std::fs::read_to_string(&self.pointer_path)
            .ok()
            .and_then(|text| serde_json::from_str::<CurrentRunManifest>(&text).ok());
        if current
            .as_ref()
            .is_some_and(|manifest| manifest.pid == self.pid)
        {
            let _ = std::fs::remove_file(&self.pointer_path);
        }
    }
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let Some(parent) = path.parent() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("path has no parent: {}", path.display()),
        ));
    };
    std::fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("current_run.json");
    let tmp_path = parent.join(format!(".{file_name}.tmp.{}", std::process::id()));
    let text = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    std::fs::write(&tmp_path, text)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

fn register_current_run(
    trade_mode: TradeMode,
    telemetry_path: Option<&PathBuf>,
    out_dir: Option<&PathBuf>,
) -> Option<CurrentRunRegistration> {
    let telemetry_path = telemetry_path?;
    let started_at_unix_ms = now_ms().max(0) as u64;
    let started_at = started_at_unix_ms.to_string();
    let manifest_path = Path::new(CURRENT_RUNS_DIR).join(format!("{}.json", std::process::id()));
    let pointer_path = PathBuf::from(CURRENT_RUN_POINTER_PATH);
    let manifest = CurrentRunManifest {
        pid: std::process::id(),
        started_at,
        started_at_unix_ms,
        trade_mode: trade_mode.as_str().to_string(),
        telemetry_path: telemetry_path.display().to_string(),
        out_dir: out_dir.map(|path| path.display().to_string()),
        manifest_path: manifest_path.display().to_string(),
    };
    if let Err(err) = write_json_atomic(&manifest_path, &manifest) {
        eprintln!(
            "paraphina_live | current_run_manifest_write_error path={} err={err}",
            manifest_path.display()
        );
        return None;
    }
    if let Err(err) = write_json_atomic(&pointer_path, &manifest) {
        eprintln!(
            "paraphina_live | current_run_pointer_write_error path={} err={err}",
            pointer_path.display()
        );
    }
    Some(CurrentRunRegistration {
        pid: manifest.pid,
        manifest_path,
        pointer_path,
    })
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn market_frontier_audit_enabled() -> bool {
    env_is_true("PARAPHINA_WS_AUDIT")
}

fn market_ingest_channel_cap() -> usize {
    env_usize("PARAPHINA_MARKET_INGEST_CHANNEL_CAP", 1024)
}

fn market_channel_cap() -> usize {
    env_usize("PARAPHINA_MARKET_CHANNEL_CAP", 1024)
}

fn connector_market_channel_cap() -> usize {
    env_usize("PARAPHINA_CONNECTOR_MARKET_CHANNEL_CAP", 1024)
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

fn connector_execution_mode_checks(
    trade_mode: TradeMode,
    connectors: &[ConnectorArg],
) -> (bool, String) {
    if !matches!(trade_mode, TradeMode::Live | TradeMode::Testnet) {
        return (true, "not_required".to_string());
    }

    let mut ok = true;
    let mut details: Vec<String> = Vec::new();
    for connector in connectors {
        match connector {
            ConnectorArg::Hyperliquid => {
                #[cfg(feature = "live_hyperliquid")]
                {
                    let cfg =
                        paraphina::live::connectors::hyperliquid::HyperliquidConfig::from_env();
                    let mode_ok = !cfg.paper_mode;
                    ok &= mode_ok;
                    details.push(format!(
                        "hyperliquid:paper_mode={} required=false",
                        cfg.paper_mode
                    ));
                }
                #[cfg(not(feature = "live_hyperliquid"))]
                {
                    ok = false;
                    details.push("hyperliquid:feature_disabled".to_string());
                }
            }
            ConnectorArg::Lighter => {
                #[cfg(feature = "live_lighter")]
                {
                    let cfg = paraphina::live::connectors::lighter::LighterConfig::from_env();
                    let mode_ok = !cfg.paper_mode;
                    ok &= mode_ok;
                    details.push(format!(
                        "lighter:paper_mode={} required=false",
                        cfg.paper_mode
                    ));
                }
                #[cfg(not(feature = "live_lighter"))]
                {
                    ok = false;
                    details.push("lighter:feature_disabled".to_string());
                }
            }
            _ => {}
        }
    }
    if details.is_empty() {
        details.push("not_applicable".to_string());
    }
    (ok, details.join(" "))
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

fn parse_reconcile_interval_ms() -> Option<u64> {
    let account_var = std::env::var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS").ok();
    let legacy_var = std::env::var("PARAPHINA_LIVE_RECONCILE_MS").ok();
    let raw = if let Some(val) = account_var {
        val
    } else if let Some(val) = legacy_var {
        eprintln!(
            "paraphina_live | warn=reconcile_env_legacy_only var=PARAPHINA_LIVE_RECONCILE_MS preferred=PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS"
        );
        val
    } else {
        return None;
    };
    let normalized = raw.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "false" | "off" | "no") {
        return None;
    }
    raw.parse::<i64>()
        .ok()
        .filter(|ms| *ms > 0)
        .map(|ms| ms as u64)
}

fn parse_live_order_snapshot_poll_ms() -> u64 {
    std::env::var("PARAPHINA_LIVE_ORDER_SNAPSHOT_POLL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2_000)
}

fn parse_aster_backstop_account_poll_ms() -> u64 {
    std::env::var("PARAPHINA_ASTER_BACKSTOP_ACCOUNT_POLL_MS")
        .ok()
        .or_else(|| std::env::var("PARAPHINA_ASTER_ACCOUNT_POLL_MS").ok())
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .unwrap_or(15_000)
}

fn paradex_private_order_truth_enabled() -> bool {
    std::env::var("PARAPHINA_PARADEX_PRIVATE_ORDER_TRUTH_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn parse_paradex_backstop_order_poll_ms() -> u64 {
    std::env::var("PARAPHINA_PARADEX_ORDER_BACKSTOP_POLL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|ms| *ms > 0)
        .unwrap_or(15_000)
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
        ConnectorArg::Lighter => {
            if cfg!(feature = "live_lighter") {
                ConnectorSupport::MarketAccountExec
            } else {
                ConnectorSupport::MissingFeature
            }
        }
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

fn connector_has_passive_fill_stream(connector: ConnectorArg) -> bool {
    matches!(connector, ConnectorArg::Hyperliquid | ConnectorArg::Lighter)
}

fn connector_has_passive_fill_visibility(connector: ConnectorArg) -> bool {
    connector_has_passive_fill_stream(connector)
        || matches!(
            connector,
            ConnectorArg::Extended | ConnectorArg::Aster | ConnectorArg::Paradex
        )
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
            if update.received_ms.is_none() {
                update.received_ms = Some(timestamp_ms);
            }
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
        paraphina::live::types::ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(mut fill) => {
            fill.venue_id = venue_id.to_string();
            fill.venue_index = venue_index;
            paraphina::live::types::ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(fill)
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
            if market_ingest_tx.send(event).await.is_err() {
                break;
            }
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

    let (exec_mode_ok, exec_mode_details) = connector_execution_mode_checks(trade_mode, connectors);
    if !exec_mode_ok {
        eprintln!(
            "paraphina_live | error=live_mode_execution_mode_invalid detail=\"{}\" (set HL_PAPER_MODE=false and LIGHTER_PAPER_MODE=false)",
            exec_mode_details
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

/// Validate config by constructing Engine + GlobalState and running one
/// synthetic tick. Returns 0 on success, 1 on failure.
fn run_validate_config(cfg: &Config) -> i32 {
    use paraphina::engine::Engine;
    use paraphina::state::GlobalState;

    println!("paraphina_live validate-config: loading config and running dry-run tick...");

    // Construct engine and state — this validates all config-derived parameters.
    let engine = Engine::new(cfg);
    let mut state = GlobalState::new(cfg);

    // Run one synthetic tick at a fixed timestamp.
    let now_ms: i64 = 1_700_000_000_000;
    engine.seed_dummy_mids(&mut state, now_ms);
    engine.main_tick(&mut state, now_ms);

    // Verify no immediate kill switch tripped on first tick.
    if state.kill_switch {
        println!(
            "FAIL: kill switch tripped on first tick (reason={:?})",
            state.kill_reason
        );
        return 1;
    }

    // Sanity: config produced valid vol precomputed values.
    if !engine.vol_pre.vol_ref_tick.is_finite() || !engine.vol_pre.sigma_min_tick.is_finite() {
        println!("FAIL: vol precomputed values are not finite");
        return 1;
    }

    // Verify fair value computation ran (may or may not be available depending
    // on venue count, but the computation itself should not panic).
    println!(
        "  risk_regime={:?} kill_switch={} fv_available={} venues={}",
        state.risk_regime,
        state.kill_switch,
        state.fv_available,
        state.venues.len()
    );
    println!("PASS: config validated successfully");
    0
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
                        Some(cfg.ws_url()),
                        Some(cfg.rest_url()),
                    );
                    let (ok, status) = endpoint_dns_status(cfg.info_url());
                    endpoint_ok &= ok;
                    endpoint_details
                        .push(format!("hyperliquid_info http={} {status}", cfg.info_url()));
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
                let api_key_index_present = env_present("LIGHTER_API_KEY_INDEX");
                let account_index_present = env_present("LIGHTER_ACCOUNT_INDEX");
                let priv_present = env_present("LIGHTER_API_PRIVATE_KEY_HEX");
                let token_present = env_present("LIGHTER_AUTH_TOKEN");
                let signer_present = env_present("LIGHTER_SIGNER_URL");
                let needs_auth = matches!(trade_mode, TradeMode::Live | TradeMode::Testnet);
                if needs_auth {
                    creds_ok = creds_ok
                        && api_key_index_present
                        && account_index_present
                        && priv_present
                        && signer_present;
                }
                let detail = format!(
                    "{}:auth_required={} api_key_index={} account_index={} api_private_key_hex={} auth_token={} signer_url_present={} fixture_dir_ok={}",
                    connector.as_str(),
                    needs_auth,
                    api_key_index_present,
                    account_index_present,
                    priv_present,
                    token_present,
                    signer_present,
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
                    let jwt_cmd_present = env_present("PARADEX_JWT_CMD");
                    let payload_present = env_present("PARADEX_AUTH_PAYLOAD_JSON");
                    let needs_auth = matches!(trade_mode, TradeMode::Live | TradeMode::Testnet);
                    if needs_auth {
                        creds_ok = creds_ok && (jwt_present || jwt_cmd_present || payload_present);
                    }
                    creds_ok = creds_ok && ws_ok && market_ok;
                    let detail = format!(
                        "{}:public_ws=true ws_url_ok={} market_ok={} jwt={} jwt_cmd={} auth_payload={}",
                        connector.as_str(),
                        ws_ok,
                        market_ok,
                        jwt_present,
                        jwt_cmd_present,
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
                    let trader_cmd_present = env_present("EXTENDED_TRADER_CMD");
                    let stark_private_present = env_present("EXTENDED_STARK_PRIVATE_KEY");
                    let stark_public_present = env_present("EXTENDED_STARK_PUBLIC_KEY");
                    let l2_vault_present = env_present("EXTENDED_L2_VAULT");
                    let needs_keys = matches!(trade_mode, TradeMode::Live | TradeMode::Testnet);
                    if needs_keys {
                        creds_ok = creds_ok
                            && key_present
                            && trader_cmd_present
                            && stark_private_present
                            && stark_public_present
                            && l2_vault_present;
                    }
                    creds_ok = creds_ok && ws_ok && market_ok;
                    let detail = format!(
                        "{}:public_ws=true ws_url_ok={} market_ok={} api_key={} trader_cmd={} stark_private={} stark_public={} l2_vault={}",
                        connector.as_str(),
                        ws_ok,
                        market_ok,
                        key_present,
                        trader_cmd_present,
                        stark_private_present,
                        stark_public_present,
                        l2_vault_present
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

    let (exec_mode_ok, exec_mode_details) = connector_execution_mode_checks(trade_mode, connectors);
    checks.push(PreflightCheck {
        label: "execution_modes",
        ok: exec_mode_ok,
        details: exec_mode_details,
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
    let telemetry_path = resolve_telemetry_path(out_dir.as_ref());
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
    export_runtime_canary_metadata(canary_profile.as_ref(), canary_settings.as_ref());
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
        apply_explicit_connector_selection_to_config(&mut cfg, &connectors);
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
    if args.validate_config {
        std::process::exit(run_validate_config(&cfg));
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
    if let Some(canary_log) =
        format_runtime_canary_log(canary_profile.as_ref(), canary_settings.as_ref())
    {
        println!("{canary_log}");
        eprintln!("{canary_log}");
    }

    let audit_dir = out_dir.clone().unwrap_or_else(default_audit_dir);
    if let Err(err) = std::fs::create_dir_all(&audit_dir) {
        eprintln!("paraphina_live | audit_dir_create_error={err}");
    }
    if let Err(err) = write_audit_files(&audit_dir, &cfg, &build_info) {
        eprintln!("paraphina_live | audit_write_error={err}");
    }
    let _current_run_registration = register_current_run(
        trade_mode.trade_mode,
        telemetry_path.as_ref(),
        out_dir.as_ref(),
    );
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

    let market_ingest_channel_cap = market_ingest_channel_cap();
    let market_channel_cap = market_channel_cap();
    let connector_market_channel_cap = connector_market_channel_cap();
    let (market_ingest_tx, mut market_ingest_rx) =
        mpsc::channel::<paraphina::live::types::MarketDataEvent>(market_ingest_channel_cap);
    let market_ingest_audit_tx = market_ingest_tx.clone();
    let (market_tx, market_rx) =
        mpsc::channel::<paraphina::live::types::MarketDataEvent>(market_channel_cap);
    let (paper_market_tx, paper_market_rx) = mpsc::channel::<PaperMarketUpdate>(1024);
    let paper_market_tx = if paper_mode {
        Some(paper_market_tx)
    } else {
        None
    };
    let override_market_ts = paper_mode && env_is_true("PARAPHINA_PAPER_USE_WALLCLOCK_TS");
    let market_frontier_audit_enabled = market_frontier_audit_enabled();
    tokio::spawn(async move {
        let mut last_emit = Instant::now();
        let mut ingest_queued_hiwater: usize = 0;
        let mut out_queued_hiwater: usize = 0;
        let mut forward_send_count: u64 = 0;
        let mut forward_send_err_count: u64 = 0;
        let mut send_block_max_ms: u64 = 0;
        let mut send_block_gt_5ms: u64 = 0;
        let mut send_block_gt_50ms: u64 = 0;
        let mut send_block_gt_250ms: u64 = 0;
        while let Some(event) = market_ingest_rx.recv().await {
            let event = if override_market_ts {
                override_market_timestamp(event, now_ms())
            } else {
                event
            };
            if let Some(tx) = paper_market_tx.as_ref() {
                if let Some(update) = paper_market_update_from_event(&event) {
                    // Non-blocking: drop paper update if channel full.
                    let _ = tx.try_send(update);
                }
            }
            let send_started = Instant::now();
            let send_result = market_tx.send(event).await;
            let send_block_ms = send_started.elapsed().as_millis() as u64;
            forward_send_count = forward_send_count.saturating_add(1);
            send_block_max_ms = send_block_max_ms.max(send_block_ms);
            if send_block_ms > 5 {
                send_block_gt_5ms = send_block_gt_5ms.saturating_add(1);
            }
            if send_block_ms > 50 {
                send_block_gt_50ms = send_block_gt_50ms.saturating_add(1);
            }
            if send_block_ms > 250 {
                send_block_gt_250ms = send_block_gt_250ms.saturating_add(1);
            }
            if send_result.is_err() {
                forward_send_err_count = forward_send_err_count.saturating_add(1);
            }
            if market_frontier_audit_enabled {
                let ingest_queued_len =
                    market_ingest_channel_cap.saturating_sub(market_ingest_audit_tx.capacity());
                let out_queued_len = market_channel_cap.saturating_sub(market_tx.capacity());
                ingest_queued_hiwater = ingest_queued_hiwater.max(ingest_queued_len);
                out_queued_hiwater = out_queued_hiwater.max(out_queued_len);
                let emit_since_ms = last_emit.elapsed().as_millis() as u64;
                if emit_since_ms >= 1_000 {
                    eprintln!(
                        "WS_AUDIT component=market_frontier reason=periodic interval_ms=1000 \
ingest_cap={} ingest_queued_len={} ingest_queued_hiwater={} out_cap={} out_queued_len={} \
out_queued_hiwater={} forward_send_count={} forward_send_err_count={} send_block_max_ms={} \
send_block_gt_5ms={} send_block_gt_50ms={} send_block_gt_250ms={} emit_since_ms={}",
                        market_ingest_channel_cap,
                        ingest_queued_len,
                        ingest_queued_hiwater,
                        market_channel_cap,
                        out_queued_len,
                        out_queued_hiwater,
                        forward_send_count,
                        forward_send_err_count,
                        send_block_max_ms,
                        send_block_gt_5ms,
                        send_block_gt_50ms,
                        send_block_gt_250ms,
                        emit_since_ms,
                    );
                    last_emit = Instant::now();
                    ingest_queued_hiwater = ingest_queued_len;
                    out_queued_hiwater = out_queued_len;
                    forward_send_count = 0;
                    forward_send_err_count = 0;
                    send_block_max_ms = 0;
                    send_block_gt_5ms = 0;
                    send_block_gt_50ms = 0;
                    send_block_gt_250ms = 0;
                }
            }
            if send_result.is_err() {
                break;
            }
        }
    });
    let (_account_tx, account_rx) = mpsc::channel::<paraphina::live::types::AccountEvent>(256);
    let (exec_tx, exec_rx) = mpsc::channel::<paraphina::live::types::ExecutionEvent>(512);
    let (_order_snapshot_tx, order_snapshot_rx) =
        mpsc::channel::<paraphina::live::types::OrderSnapshot>(128);
    let (account_reconcile_tx, account_reconcile_rx) = mpsc::channel::<LiveAccountRequest>(32);
    let mut account_refresh_handlers: HashMap<usize, AccountRefreshHandler> = HashMap::new();

    if let Some(ms) = parse_reconcile_interval_ms() {
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
    // Keep the emergency single-flight lane as deep as the main order queue so
    // cleanup and inventory-brake requests do not drop behind closeout bursts.
    let (priority_order_tx, mut priority_order_rx) = mpsc::channel::<LiveOrderRequest>(256);
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
        let (market_tx, market_rx) = mpsc::channel(connector_market_channel_cap);
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

    // SharedVenueAges: created early so enforcer + REST monitor closures can reference it.
    let shared_venue_ages =
        paraphina::live::shared_venue_ages::SharedVenueAges::new(cfg.venues.len());
    let mut exec_clients: BTreeMap<String, Arc<dyn LiveRestClient>> = BTreeMap::new();
    // Layer A + B: collect slots for the health enforcer and REST monitor.
    let mut enforcer_slots: Vec<paraphina::live::venue_health_enforcer::ConnectorSlot> = Vec::new();
    let mut rest_entries: Vec<paraphina::live::rest_health_monitor::VenueRestEntry> = Vec::new();
    if trade_mode.trade_mode != TradeMode::Shadow {
        let degraded = connectors
            .iter()
            .copied()
            .filter(|connector| {
                matches!(
                    connector_support(*connector),
                    ConnectorSupport::MarketAccountExec
                ) && !connector_has_passive_fill_visibility(*connector)
            })
            .map(|connector| connector.as_str())
            .collect::<Vec<_>>();
        if !degraded.is_empty() {
            eprintln!(
                "paraphina_live | passive_fill_visibility_degraded=true connectors={} reason=no_fill_stream_or_snapshot_visibility",
                degraded.join(",")
            );
        }
    }
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
                    if trade_mode.trade_mode != TradeMode::Shadow && hl_cfg.vault_address.is_some()
                    {
                        let refresh_client = hl_arc.clone();
                        register_account_refresh_handler(
                            &mut account_refresh_handlers,
                            "hyperliquid",
                            venue_id.clone(),
                            venue_index,
                            Arc::new(move || {
                                let client = refresh_client.clone();
                                Box::pin(async move {
                                    match client.fetch_account_snapshot().await {
                                        Ok(paraphina::live::types::AccountEvent::Snapshot(
                                            snapshot,
                                        )) => Ok(snapshot),
                                        Err(err) => Err(
                                            paraphina::live::gateway::LiveGatewayError::retryable(
                                                format!(
                                                    "hyperliquid account snapshot error: {err}"
                                                ),
                                            ),
                                        ),
                                    }
                                })
                            }),
                        );
                    }
                    if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                        if hl_cfg.paper_mode {
                            eprintln!(
                                "paraphina_live | exec_disabled=true reason=hl_paper_mode connector=hyperliquid"
                            );
                        } else if hl_cfg.private_key_hex.is_some() {
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
                        spawn_supervised("hyperliquid_account_poll", move || {
                            let hl = hl_poll.clone();
                            async move { hl.run_account_polling(poll_ms).await }
                        });
                        if hl_cfg.private_key_hex.is_some() {
                            let hl_private = hl_arc.clone();
                            spawn_supervised("hyperliquid_private_ws", move || {
                                let hl = hl_private.clone();
                                async move { hl.run_private_ws().await }
                            });
                        } else {
                            eprintln!("paraphina_live | private_ws_disabled=true reason=missing_hl_private_key connector=hyperliquid");
                        }
                    }
                    if trade_mode.trade_mode != TradeMode::Shadow
                        && hl_cfg.private_key_hex.is_some()
                        && hl_arc.uses_ws_post_actions()
                    {
                        eprintln!(
                            "paraphina_live | connector=hyperliquid | action_transport={}",
                            hl_arc.action_transport_label()
                        );
                        let hl_post = hl_arc.clone();
                        spawn_supervised("hyperliquid_post_ws", move || {
                            let hl = hl_post.clone();
                            async move { hl.run_post_ws().await }
                        });
                    }
                    let hl_public = hl_arc.clone();
                    let hl_handle = spawn_supervised("hyperliquid_public_ws", move || {
                        let hl = hl_public.clone();
                        async move { hl.run_public_ws().await }
                    });
                    // Layer A: enforcer slot for force-restart.
                    {
                        let hl_respawn = hl_arc.clone();
                        enforcer_slots.push(
                            paraphina::live::venue_health_enforcer::ConnectorSlot {
                                name: "hyperliquid_public_ws".to_string(),
                                venue_index,
                                handle: hl_handle,
                                respawn: Box::new(move || {
                                    let hl = hl_respawn.clone();
                                    spawn_supervised("hyperliquid_public_ws", move || {
                                        let h = hl.clone();
                                        async move { h.run_public_ws().await }
                                    })
                                }),
                                last_abort: None,
                            },
                        );
                    }
                    // Layer B: REST monitor entry for Hyperliquid.
                    {
                        let hl_rest_cfg = hl_cfg.clone();
                        let hl_rest_http = reqwest::Client::builder()
                            .timeout(std::time::Duration::from_secs(5))
                            .build()
                            .expect("hl rest http client");
                        rest_entries.push(paraphina::live::rest_health_monitor::VenueRestEntry {
                            name: "hyperliquid".to_string(),
                            venue_index,
                            mode: paraphina::live::rest_health_monitor::RestMonitorMode::InjectSnapshot,
                            fetcher: Box::new(move || {
                                let c = hl_rest_cfg.clone();
                                let h = hl_rest_http.clone();
                                let url = c.info_url().to_string();
                                Box::pin(async move {
                                    paraphina::live::connectors::hyperliquid::fetch_l2_snapshot(
                                        &h, &c, &url,
                                    )
                                    .await
                                    .map(paraphina::live::rest_health_monitor::RestFetchOutcome::Inject)
                                })
                            }),
                        });
                    }
                    let hl_funding = hl_arc.clone();
                    let funding_poll_ms = std::env::var("HL_FUNDING_POLL_MS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(10_000);
                    spawn_supervised("hyperliquid_funding_poll", move || {
                        let hl = hl_funding.clone();
                        async move { hl.run_funding_polling(funding_poll_ms).await }
                    });
                    let hl_rest_fallback = hl_arc.clone();
                    spawn_supervised("hyperliquid_rest_book_fallback", move || {
                        let hl = hl_rest_fallback.clone();
                        async move { hl.run_rest_book_fallback().await }
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
                    let fixture_dir = std::env::var("LIGHTER_FIXTURE_DIR")
                        .ok()
                        .map(std::path::PathBuf::from);
                    let use_fixture = fixture_dir.is_some();
                    if trade_mode.trade_mode != TradeMode::Shadow {
                        if lighter_cfg.paper_mode {
                            eprintln!("paraphina_live | account_snapshots_disabled=true reason=lighter_paper_mode connector=lighter");
                            if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                            }
                        } else if use_fixture || lighter_cfg.has_auth() {
                            lighter = lighter.with_account_tx(channels.account_tx.clone());
                        } else {
                            eprintln!("paraphina_live | account_snapshots_disabled=true reason=missing_lighter_auth connector=lighter");
                            if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                                send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                            }
                        }
                    } else {
                        eprintln!("paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=lighter");
                        if let Some(index) = resolve_venue_index(&cfg, &venue_id) {
                            send_unavailable_account_snapshot_for(&_account_tx, &cfg, index);
                        }
                    }
                    let lighter_arc = Arc::new(lighter);
                    if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
                        if use_fixture {
                            eprintln!("paraphina_live | exec_disabled=true reason=lighter_fixture_mode connector=lighter");
                        } else if lighter_cfg.paper_mode {
                            eprintln!("paraphina_live | exec_disabled=true reason=lighter_paper_mode connector=lighter");
                        } else if lighter_cfg.has_auth()
                            && lighter_cfg
                                .signer_url
                                .as_ref()
                                .map(|s| !s.trim().is_empty())
                                .unwrap_or(false)
                        {
                            exec_clients.insert(venue_id.clone(), lighter_arc.clone());
                        } else {
                            if lighter_cfg.has_auth() {
                                eprintln!("paraphina_live | exec_disabled=true reason=missing_lighter_signer connector=lighter");
                            } else {
                                eprintln!("paraphina_live | exec_disabled=true reason=missing_lighter_auth connector=lighter");
                            }
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
                        let ltr_rest_url = lighter_cfg.rest_url.clone();
                        let ltr_market = lighter_cfg.market.clone();
                        if trade_mode.trade_mode != TradeMode::Shadow
                            && !lighter_cfg.paper_mode
                            && !use_fixture
                            && lighter_cfg.has_auth()
                        {
                            let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                                .ok()
                                .and_then(|v| v.parse::<u64>().ok())
                                .unwrap_or(5_000);
                            let poll_ms =
                                paraphina::live::connectors::lighter::lighter_account_poll_ms(
                                    poll_ms,
                                );
                            let lighter_poll = lighter_arc.clone();
                            spawn_supervised("lighter_account_poll", move || {
                                let l = lighter_poll.clone();
                                async move { l.run_account_polling(poll_ms).await }
                            });
                        }
                        let lighter_public = lighter_arc.clone();
                        let lighter_handle = spawn_supervised("lighter_public_ws", move || {
                            let l = lighter_public.clone();
                            async move { l.run_public_ws().await }
                        });
                        // Layer A: enforcer slot.
                        {
                            let lighter_respawn = lighter_arc.clone();
                            enforcer_slots.push(
                                paraphina::live::venue_health_enforcer::ConnectorSlot {
                                    name: "lighter_public_ws".to_string(),
                                    venue_index,
                                    handle: lighter_handle,
                                    respawn: Box::new(move || {
                                        let l = lighter_respawn.clone();
                                        spawn_supervised("lighter_public_ws", move || {
                                            let l2 = l.clone();
                                            async move { l2.run_public_ws().await }
                                        })
                                    }),
                                    last_abort: None,
                                },
                            );
                        }
                        // Layer B: REST monitor entry for Lighter.
                        {
                            let rest_url = ltr_rest_url;
                            let market = ltr_market;
                            let vi = venue_index;
                            let ltr_http = reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(5))
                                .build()
                                .expect("lighter rest http client");
                            rest_entries.push(paraphina::live::rest_health_monitor::VenueRestEntry {
                                name: "lighter".to_string(),
                                venue_index: vi,
                                mode: paraphina::live::rest_health_monitor::RestMonitorMode::InjectSnapshot,
                                fetcher: Box::new(move || {
                                    let h = ltr_http.clone();
                                    let ru = rest_url.clone();
                                    let m = market.clone();
                                    Box::pin(async move {
                                        paraphina::live::rest_health_monitor::fetch_lighter_l2_snapshot(
                                            &h, &ru, &m, vi,
                                        )
                                        .await
                                    })
                                }),
                            });
                        }
                        let lighter_funding = lighter_arc.clone();
                        let funding_poll_ms = std::env::var("LIGHTER_FUNDING_POLL_MS")
                            .ok()
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(10_000);
                        spawn_supervised("lighter_funding_poll", move || {
                            let l = lighter_funding.clone();
                            async move { l.run_funding_polling(funding_poll_ms).await }
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
                        // Capture REST monitor values before extended_cfg is moved.
                        let ext_rest_url = extended_cfg.rest_url.clone();
                        let ext_market = extended_cfg.market.clone();
                        let ext_depth_limit = extended_cfg.depth_limit;
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
                        let ext_handle = spawn_supervised("extended_public_ws", move || {
                            let e = extended_public.clone();
                            async move { e.run_public_ws().await }
                        });
                        // Layer A: enforcer slot.
                        {
                            let ext_respawn = extended_arc.clone();
                            enforcer_slots.push(
                                paraphina::live::venue_health_enforcer::ConnectorSlot {
                                    name: "extended_public_ws".to_string(),
                                    venue_index,
                                    handle: ext_handle,
                                    respawn: Box::new(move || {
                                        let e = ext_respawn.clone();
                                        spawn_supervised("extended_public_ws", move || {
                                            let e2 = e.clone();
                                            async move { e2.run_public_ws().await }
                                        })
                                    }),
                                    last_abort: None,
                                },
                            );
                        }
                        // Layer B: REST monitor entry.
                        {
                            let rest_url = ext_rest_url;
                            let market = ext_market;
                            let depth_limit = ext_depth_limit;
                            let vi = venue_index;
                            let ext_http = reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(5))
                                .build()
                                .expect("ext rest http client");
                            rest_entries.push(paraphina::live::rest_health_monitor::VenueRestEntry {
                                name: "extended".to_string(),
                                venue_index: vi,
                                mode: paraphina::live::rest_health_monitor::RestMonitorMode::InjectSnapshot,
                                fetcher: Box::new(move || {
                                    let h = ext_http.clone();
                                    let ru = rest_url.clone();
                                    let m = market.clone();
                                    Box::pin(async move {
                                        paraphina::live::rest_health_monitor::fetch_extended_l2_snapshot(
                                            &h, &ru, &m, depth_limit, vi,
                                        )
                                        .await
                                    })
                                }),
                            });
                        }
                        let extended_funding = extended_arc.clone();
                        let funding_poll_ms = std::env::var("EXTENDED_FUNDING_POLL_MS")
                            .ok()
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(10_000);
                        spawn_supervised("extended_funding_poll", move || {
                            let e = extended_funding.clone();
                            async move { e.run_funding_polling(funding_poll_ms).await }
                        });
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_private_read_auth() {
                                let account_tx = channels.account_tx.clone();
                                let exec_tx = channels.exec_tx.clone();
                                let private_venue_id = venue_id.clone();
                                let rest_client_clone = rest_client.clone();
                                tokio::spawn(async move {
                                    rest_client_clone
                                        .run_private_ws(
                                            account_tx,
                                            exec_tx,
                                            private_venue_id,
                                            venue_index,
                                        )
                                        .await;
                                });
                                if rest_client.has_execution_auth() {
                                    let refresh_client = rest_client.clone();
                                    let refresh_venue_id = venue_id.clone();
                                    register_account_refresh_handler(
                                        &mut account_refresh_handlers,
                                        "extended",
                                        venue_id.clone(),
                                        venue_index,
                                        Arc::new(move || {
                                            let client = refresh_client.clone();
                                            let venue_id = refresh_venue_id.clone();
                                            Box::pin(async move {
                                                client
                                                    .fetch_account_snapshot(&venue_id, venue_index)
                                                    .await
                                            })
                                        }),
                                    );
                                    let poll_ms = std::env::var(
                                        "PARAPHINA_EXTENDED_ACCOUNT_BACKSTOP_POLL_MS",
                                    )
                                    .ok()
                                    .and_then(|v| v.parse::<u64>().ok())
                                    .unwrap_or(15_000);
                                    let account_tx = channels.account_tx.clone();
                                    let account_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_account_polling(
                                                account_tx,
                                                account_venue_id,
                                                venue_index,
                                                poll_ms,
                                            )
                                            .await;
                                    });
                                    let order_poll_ms =
                                        std::env::var("PARAPHINA_EXTENDED_ORDER_BACKSTOP_POLL_MS")
                                            .ok()
                                            .and_then(|v| v.parse::<u64>().ok())
                                            .unwrap_or(15_000);
                                    let exec_tx = channels.exec_tx.clone();
                                    let order_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_order_polling(
                                                exec_tx,
                                                order_venue_id,
                                                venue_index,
                                                order_poll_ms,
                                            )
                                            .await;
                                    });
                                } else {
                                    eprintln!(
                                        "paraphina_live | account_backstop_disabled=true reason=missing_extended_bridge_auth connector=extended"
                                    );
                                }
                            } else {
                                eprintln!(
                                    "paraphina_live | account_snapshots_disabled=true reason=missing_extended_api_key connector=extended"
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
                            if rest_client.has_execution_auth() {
                                exec_clients.insert(venue_id.clone(), rest_client.clone());
                            } else {
                                eprintln!(
                                    "paraphina_live | exec_disabled=true reason=missing_extended_bridge_auth connector=extended"
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
                        // Capture REST monitor values before aster_cfg is moved.
                        let ast_rest_url = aster_cfg.rest_url.clone();
                        let ast_market = aster_cfg.market.clone();
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
                        let aster_handle = spawn_supervised("aster_public_ws", move || {
                            let a = aster_public.clone();
                            async move { a.run_public_ws().await }
                        });
                        // Layer A: enforcer slot.
                        {
                            let aster_respawn = aster_arc.clone();
                            enforcer_slots.push(
                                paraphina::live::venue_health_enforcer::ConnectorSlot {
                                    name: "aster_public_ws".to_string(),
                                    venue_index,
                                    handle: aster_handle,
                                    respawn: Box::new(move || {
                                        let a = aster_respawn.clone();
                                        spawn_supervised("aster_public_ws", move || {
                                            let a2 = a.clone();
                                            async move { a2.run_public_ws().await }
                                        })
                                    }),
                                    last_abort: None,
                                },
                            );
                        }
                        // Layer B: REST monitor entry.
                        {
                            let rest_url = ast_rest_url;
                            let market = ast_market;
                            let vi = venue_index;
                            let aster_budget = aster_arc.public_rest_budget();
                            let aster_http = reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(5))
                                .build()
                                .expect("aster rest http client");
                            rest_entries.push(paraphina::live::rest_health_monitor::VenueRestEntry {
                                name: "aster".to_string(),
                                venue_index: vi,
                                mode: paraphina::live::rest_health_monitor::RestMonitorMode::ProbeOnly,
                                fetcher: Box::new(move || {
                                    let h = aster_http.clone();
                                    let ru = rest_url.clone();
                                    let m = market.clone();
                                    let budget = aster_budget.clone();
                                    Box::pin(async move {
                                        paraphina::live::rest_health_monitor::probe_aster_book_ticker_budgeted(
                                            &h, &ru, &m, &budget,
                                        )
                                        .await
                                    })
                                }),
                            });
                        }
                        let aster_funding = aster_arc.clone();
                        spawn_supervised("aster_mark_price_ws", move || {
                            let a = aster_funding.clone();
                            async move { a.run_mark_price_ws().await }
                        });
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                let refresh_client = rest_client.clone();
                                let refresh_venue_id = venue_id.clone();
                                register_account_refresh_handler(
                                    &mut account_refresh_handlers,
                                    "aster",
                                    venue_id.clone(),
                                    venue_index,
                                    Arc::new(move || {
                                        let client = refresh_client.clone();
                                        let venue_id = refresh_venue_id.clone();
                                        Box::pin(async move {
                                            client
                                                .fetch_account_snapshot(&venue_id, venue_index)
                                                .await
                                        })
                                    }),
                                );
                                let use_user_stream = std::env::var("PARAPHINA_ASTER_USER_STREAM")
                                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                                    .unwrap_or(false);
                                if use_user_stream {
                                    let account_tx = channels.account_tx.clone();
                                    let exec_tx = channels.exec_tx.clone();
                                    let account_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    spawn_supervised("aster_private_ws", move || {
                                        let rest = rest_client_clone.clone();
                                        let account_tx = account_tx.clone();
                                        let exec_tx = exec_tx.clone();
                                        let venue_id = account_venue_id.clone();
                                        async move {
                                            rest.run_private_ws(
                                                account_tx,
                                                exec_tx,
                                                venue_id,
                                                venue_index,
                                            )
                                            .await
                                        }
                                    });
                                    let poll_ms = parse_aster_backstop_account_poll_ms();
                                    eprintln!(
                                        "paraphina_live | aster_account_backstop_poll_ms={} mode=user_stream",
                                        poll_ms
                                    );
                                    let account_tx = channels.account_tx.clone();
                                    let account_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_account_polling(
                                                account_tx,
                                                account_venue_id,
                                                venue_index,
                                                poll_ms,
                                            )
                                            .await;
                                    });
                                } else {
                                    let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                                        .ok()
                                        .and_then(|v| v.parse::<u64>().ok())
                                        .unwrap_or(5_000);
                                    let account_tx = channels.account_tx.clone();
                                    let account_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_account_polling(
                                                account_tx,
                                                account_venue_id,
                                                venue_index,
                                                poll_ms,
                                            )
                                            .await;
                                    });
                                    let order_poll_ms = parse_live_order_snapshot_poll_ms();
                                    let exec_tx = channels.exec_tx.clone();
                                    let order_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_order_polling(
                                                exec_tx,
                                                order_venue_id,
                                                venue_index,
                                                order_poll_ms,
                                            )
                                            .await;
                                    });
                                }
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
                        // Capture REST monitor values before paradex_cfg is moved.
                        let pdx_rest_url = paradex_cfg.rest_url.clone();
                        let pdx_market = paradex_cfg.market.clone();
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
                        let paradex_handle = spawn_supervised("paradex_public_ws", move || {
                            let p = paradex_public.clone();
                            async move { p.run_public_ws().await }
                        });
                        // Layer A: enforcer slot.
                        {
                            let pdx_respawn = paradex_arc.clone();
                            enforcer_slots.push(
                                paraphina::live::venue_health_enforcer::ConnectorSlot {
                                    name: "paradex_public_ws".to_string(),
                                    venue_index,
                                    handle: paradex_handle,
                                    respawn: Box::new(move || {
                                        let p = pdx_respawn.clone();
                                        spawn_supervised("paradex_public_ws", move || {
                                            let p2 = p.clone();
                                            async move { p2.run_public_ws().await }
                                        })
                                    }),
                                    last_abort: None,
                                },
                            );
                        }
                        // Layer B: REST monitor entry for Paradex.
                        {
                            let rest_url = pdx_rest_url;
                            let market = pdx_market;
                            let vi = venue_index;
                            let pdx_http = reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(5))
                                .build()
                                .expect("paradex rest http client");
                            rest_entries.push(paraphina::live::rest_health_monitor::VenueRestEntry {
                                name: "paradex".to_string(),
                                venue_index: vi,
                                mode: paraphina::live::rest_health_monitor::RestMonitorMode::InjectSnapshot,
                                fetcher: Box::new(move || {
                                    let h = pdx_http.clone();
                                    let ru = rest_url.clone();
                                    let m = market.clone();
                                    Box::pin(async move {
                                        paraphina::live::rest_health_monitor::fetch_paradex_l2_snapshot(
                                            &h, &ru, &m, 20, vi,
                                        )
                                        .await
                                    })
                                }),
                            });
                        }
                        let paradex_funding = paradex_arc.clone();
                        let funding_poll_ms = std::env::var("PARADEX_FUNDING_POLL_MS")
                            .ok()
                            .and_then(|v| v.parse::<u64>().ok())
                            .unwrap_or(10_000);
                        spawn_supervised("paradex_funding_poll", move || {
                            let p = paradex_funding.clone();
                            async move { p.run_funding_polling(funding_poll_ms).await }
                        });
                        if trade_mode.trade_mode != TradeMode::Shadow {
                            if rest_client.has_auth() {
                                let refresh_client = rest_client.clone();
                                let refresh_venue_id = venue_id.clone();
                                register_account_refresh_handler(
                                    &mut account_refresh_handlers,
                                    "paradex",
                                    venue_id.clone(),
                                    venue_index,
                                    Arc::new(move || {
                                        let client = refresh_client.clone();
                                        let venue_id = refresh_venue_id.clone();
                                        Box::pin(async move {
                                            client
                                                .fetch_account_snapshot(&venue_id, venue_index)
                                                .await
                                        })
                                    }),
                                );
                                let poll_ms = std::env::var("PARAPHINA_LIVE_ACCOUNT_POLL_MS")
                                    .ok()
                                    .and_then(|v| v.parse::<u64>().ok())
                                    .unwrap_or(5_000);
                                let account_tx = channels.account_tx.clone();
                                let account_venue_id = venue_id.clone();
                                let rest_client_clone = rest_client.clone();
                                tokio::spawn(async move {
                                    rest_client_clone
                                        .run_account_polling(
                                            account_tx,
                                            account_venue_id,
                                            venue_index,
                                            poll_ms,
                                        )
                                        .await;
                                });
                                if paradex_private_order_truth_enabled() {
                                    let exec_tx = channels.exec_tx.clone();
                                    let private_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    spawn_supervised("paradex_private_order_ws", move || {
                                        let rest = rest_client_clone.clone();
                                        let exec_tx = exec_tx.clone();
                                        let venue_id = private_venue_id.clone();
                                        async move {
                                            rest.run_private_order_ws(
                                                exec_tx,
                                                venue_id,
                                                venue_index,
                                            )
                                            .await
                                        }
                                    });
                                    let order_poll_ms = parse_paradex_backstop_order_poll_ms();
                                    eprintln!(
                                        "paraphina_live | paradex_order_backstop_poll_ms={} mode=private_order_truth",
                                        order_poll_ms
                                    );
                                    let exec_tx = channels.exec_tx.clone();
                                    let order_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_order_polling(
                                                exec_tx,
                                                order_venue_id,
                                                venue_index,
                                                order_poll_ms,
                                            )
                                            .await;
                                    });
                                } else {
                                    let order_poll_ms = parse_live_order_snapshot_poll_ms();
                                    let exec_tx = channels.exec_tx.clone();
                                    let order_venue_id = venue_id.clone();
                                    let rest_client_clone = rest_client.clone();
                                    tokio::spawn(async move {
                                        rest_client_clone
                                            .run_order_polling(
                                                exec_tx,
                                                order_venue_id,
                                                venue_index,
                                                order_poll_ms,
                                            )
                                            .await;
                                    });
                                }
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

    let account_reconcile_tx_for_runner = if account_refresh_handlers.is_empty() {
        None
    } else {
        spawn_account_refresh_router(account_reconcile_rx, account_refresh_handlers);
        Some(account_reconcile_tx.clone())
    };

    // ── Layer A: spawn the venue health enforcer ──────────────────────────
    // Spawned as a plain tokio task because ConnectorSlot contains non-Clone
    // JoinHandles.  The enforcer loops forever; if it panics we lose
    // enforcement but connectors still have their own internal supervision.
    if !enforcer_slots.is_empty() {
        let enforcer_ages = shared_venue_ages.clone();
        tokio::spawn(async move {
            paraphina::live::venue_health_enforcer::run_venue_health_enforcer(
                enforcer_ages,
                enforcer_slots,
                paraphina::live::venue_health_enforcer::EnforcerConfig::default(),
            )
            .await;
        });
    }

    // ── Layer B: spawn the central REST health monitor ──────────────────
    // Uses market_ingest_tx so events pass through the standard pipeline
    // (timestamp override for paper mode, paper market updates, etc.)
    if !rest_entries.is_empty() {
        let rest_ages = shared_venue_ages.clone();
        let rest_market_tx = market_ingest_tx.clone();
        tokio::spawn(async move {
            paraphina::live::rest_health_monitor::run_rest_health_monitor(
                rest_ages,
                rest_entries,
                rest_market_tx,
                paraphina::live::rest_health_monitor::RestMonitorConfig::default(),
            )
            .await;
        });
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
        let mut pending_order_reqs = VecDeque::new();
        let mut pending_priority_order_reqs = VecDeque::new();
        loop {
            tokio::select! {
                biased;
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
                Some(req) = async {
                    if pending_priority_order_reqs.is_empty() {
                        if let Some(req) = priority_order_rx.recv().await {
                            pending_priority_order_reqs.push_back(req);
                        }
                    }
                    take_priority_request(&mut pending_priority_order_reqs, &mut priority_order_rx)
                } => {
                    let LiveOrderRequest {
                        intents,
                        action_batch,
                        now_ms,
                        transport_hint,
                        response,
                    } = req;
                    let events = if let Some(gateway) = live_gateway.as_mut() {
                        handle_live_gateway_intents(
                            gateway,
                            intents,
                            action_batch.tick_index,
                            now_ms,
                            &mut exec_seq,
                            false,
                            transport_hint,
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
                    respond_to_order_request(response, events, &exec_tx);
                }
                Some(req) = async {
                    if let Some(req) = pending_order_reqs.pop_front() {
                        Some(req)
                    } else {
                        order_rx.recv().await
                    }
                } => {
                    let normalize_hyperliquid_batch_window =
                        request_contains_hyperliquid_batchable_intents(&req);
                    let req = if normalize_hyperliquid_batch_window {
                        coalesce_hyperliquid_fire_and_forget_request(
                            req,
                            &mut pending_order_reqs,
                            &mut order_rx,
                            &mut pending_priority_order_reqs,
                            &mut priority_order_rx,
                            Duration::from_millis(HYPERLIQUID_BATCH_WINDOW_MS),
                        )
                        .await
                    } else {
                        coalesce_fire_and_forget_request(
                            req,
                            &mut pending_order_reqs,
                            &mut order_rx,
                        )
                    };
                    let LiveOrderRequest {
                        intents,
                        action_batch,
                        now_ms,
                        transport_hint,
                        response,
                    } = req;
                    let events = if let Some(gateway) = live_gateway.as_mut() {
                        handle_live_gateway_intents(
                            gateway,
                            intents,
                            action_batch.tick_index,
                            now_ms,
                            &mut exec_seq,
                            normalize_hyperliquid_batch_window,
                            transport_hint,
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
                    respond_to_order_request(response, events, &exec_tx);
                }
                else => break,
            }
        }
    });

    let channels = LiveChannels {
        market_rx,
        account_rx,
        exec_rx: Some(exec_rx),
        account_reconcile_tx: account_reconcile_tx_for_runner,
        priority_order_tx,
        order_tx,
        order_snapshot_rx: Some(order_snapshot_rx),
        shared_venue_ages: Some(shared_venue_ages.clone()),
    };
    let max_orders_per_tick = std::env::var("PARAPHINA_LIVE_TELEMETRY_MAX_ORDERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(200);
    let telemetry_cfg = TelemetryConfig {
        mode: TelemetryMode::from_env(),
        path: telemetry_path.clone(),
        append: TelemetryConfig::append_from_env(),
    };
    let telemetry_sink = TelemetrySink::from_config(telemetry_cfg);
    let telemetry = LiveTelemetry {
        sink: TelemetrySinkHandle::Async(telemetry_sink.into_async()),
        shadow_mode: trade_mode.trade_mode == TradeMode::Shadow,
        execution_mode: trade_mode.trade_mode.as_str(),
        max_orders_per_tick,
        stats: Arc::new(LiveTelemetryStats::default()),
    };
    let hooks = LiveRuntimeHooks {
        metrics,
        health,
        telemetry: Some(telemetry.clone()),
    };
    let max_ticks = std::env::var("PARAPHINA_LIVE_MAX_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let run_mode = LiveRunMode::Realtime {
        interval_ms: cfg.main_loop_interval_ms as u64,
        max_ticks,
    };
    let summary = run_live_loop(&cfg, channels, run_mode, Some(hooks)).await;

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

    if should_fail_on_unexpected_live_loop_exit(run_mode) {
        eprintln!(
            "paraphina_live | error=unexpected_live_loop_exit trade_mode={} ticks_run={} kill_switch={} ready_market_count={} stale_market_count={} fv_available={}",
            trade_mode.trade_mode.as_str(),
            summary.ticks_run,
            summary.kill_switch,
            summary.ready_market_count,
            summary.stale_market_count,
            summary.fv_available,
        );
        std::process::exit(1);
    }
}

async fn handle_live_gateway_intents<C: LiveRestClient>(
    gateway: &mut LiveGateway<C>,
    intents: Vec<paraphina::types::OrderIntent>,
    tick: u64,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
    normalize_hyperliquid_batch_window: bool,
    transport_hint: TransportHint,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    let expanded = expand_live_gateway_intents(intents);
    let expanded = if normalize_hyperliquid_batch_window {
        normalize_hyperliquid_batch_window_expanded_intents(expanded)
    } else {
        expanded
    };
    if expanded.is_empty() {
        return Vec::new();
    }
    let mut results = gateway
        .submit_batch(&expanded, tick, now_ms, transport_hint)
        .await;
    if results.len() != expanded.len() {
        let err =
            paraphina::live::gateway::LiveGatewayError::fatal("live_gateway_batch_len_mismatch");
        results = vec![Err(err); expanded.len()];
    }
    let mut events = Vec::new();
    for (intent, result) in expanded.into_iter().zip(results.into_iter()) {
        let mut out = execution_events_from_gateway_result(intent, result, now_ms, seq);
        events.append(&mut out);
    }
    events
}

fn expand_live_gateway_intents(
    intents: Vec<paraphina::types::OrderIntent>,
) -> Vec<paraphina::types::OrderIntent> {
    let mut expanded = Vec::with_capacity(intents.len() * 2);
    for intent in intents {
        match intent {
            paraphina::types::OrderIntent::Replace(replace)
                if preserve_native_replace_intent(&replace) =>
            {
                expanded.push(paraphina::types::OrderIntent::Replace(replace));
            }
            paraphina::types::OrderIntent::Replace(replace) => {
                expanded.push(paraphina::types::OrderIntent::Cancel(
                    paraphina::types::CancelOrderIntent {
                        venue_index: replace.venue_index,
                        venue_id: replace.venue_id.clone(),
                        order_id: replace.order_id.clone(),
                    },
                ));
                expanded.push(paraphina::types::OrderIntent::Place(
                    paraphina::types::PlaceOrderIntent {
                        venue_index: replace.venue_index,
                        venue_id: replace.venue_id,
                        side: replace.side,
                        price: replace.price,
                        size: replace.size,
                        purpose: replace.purpose,
                        time_in_force: replace.time_in_force,
                        post_only: replace.post_only,
                        reduce_only: replace.reduce_only,
                        client_order_id: replace.client_order_id,
                        phase51_target_key: replace.phase51_target_key,
                    },
                ));
            }
            other => expanded.push(other),
        }
    }
    expanded
}

fn preserve_native_replace_intent(replace: &paraphina::types::ReplaceOrderIntent) -> bool {
    if replace.reduce_only
        || !replace.post_only
        || replace.purpose != paraphina::types::OrderPurpose::Mm
    {
        return false;
    }
    if replace
        .venue_id
        .as_ref()
        .eq_ignore_ascii_case("hyperliquid")
    {
        return is_hyperliquid_replace_identity(&replace.order_id);
    }
    if replace.venue_id.as_ref().eq_ignore_ascii_case("paradex") {
        return !replace.order_id.starts_with("co_");
    }
    if replace.venue_id.as_ref().eq_ignore_ascii_case("lighter") {
        return is_lighter_replace_identity(&replace.order_id);
    }
    if replace.venue_id.as_ref().eq_ignore_ascii_case("extended") {
        return env_is_true("PARAPHINA_EXTENDED_NATIVE_REPLACE_ENABLED")
            && is_extended_replace_identity(&replace.order_id);
    }
    false
}

const HYPERLIQUID_BATCH_WINDOW_MS: u64 = 100;

fn is_hyperliquid_cloid(order_id: &str) -> bool {
    order_id.len() == 34
        && order_id.starts_with("0x")
        && order_id[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_hyperliquid_replace_identity(order_id: &str) -> bool {
    is_hyperliquid_cloid(order_id) || order_id.parse::<u64>().is_ok()
}

fn is_lighter_replace_identity(order_id: &str) -> bool {
    order_id.parse::<u64>().is_ok()
}

fn is_extended_replace_identity(order_id: &str) -> bool {
    let order_id = order_id.trim();
    !order_id.is_empty() && !order_id.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_hyperliquid_batchable_intent(intent: &paraphina::types::OrderIntent) -> bool {
    match intent {
        paraphina::types::OrderIntent::Cancel(cancel) => {
            cancel.venue_id.eq_ignore_ascii_case("hyperliquid")
        }
        paraphina::types::OrderIntent::Place(place) => {
            place.venue_id.eq_ignore_ascii_case("hyperliquid") && place.post_only
        }
        paraphina::types::OrderIntent::Replace(replace) => {
            replace.venue_id.eq_ignore_ascii_case("hyperliquid") && replace.post_only
        }
        paraphina::types::OrderIntent::CancelAll(_) => false,
    }
}

fn request_contains_hyperliquid_batchable_intents(request: &LiveOrderRequest) -> bool {
    matches!(request.response, ResponseMode::FireAndForget)
        && request.intents.iter().any(is_hyperliquid_batchable_intent)
}

fn merge_fire_and_forget_request(request: &mut LiveOrderRequest, next: LiveOrderRequest) -> usize {
    let next_len = next.intents.len();
    request.intents.extend(next.intents);
    request.action_batch = next.action_batch;
    request.now_ms = next.now_ms;
    next_len
}

fn hyperliquid_mm_batch_same_side_dedup_enabled() -> bool {
    env_is_true("PARAPHINA_HL_MM_BATCH_SAME_SIDE_DEDUP")
}

fn hyperliquid_mm_same_side_place_key(
    intent: &paraphina::types::OrderIntent,
) -> Option<(usize, paraphina::types::Side)> {
    match intent {
        paraphina::types::OrderIntent::Place(place)
            if place.venue_id.eq_ignore_ascii_case("hyperliquid")
                && place.post_only
                && !place.reduce_only
                && place.purpose == paraphina::types::OrderPurpose::Mm =>
        {
            Some((place.venue_index, place.side))
        }
        _ => None,
    }
}

fn dedup_hyperliquid_mm_same_side_places(
    intents: Vec<paraphina::types::OrderIntent>,
) -> Vec<paraphina::types::OrderIntent> {
    let mut seen: Vec<(usize, paraphina::types::Side)> = Vec::new();
    let mut retained = Vec::with_capacity(intents.len());
    for intent in intents.into_iter().rev() {
        if let Some(key) = hyperliquid_mm_same_side_place_key(&intent) {
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
        }
        retained.push(intent);
    }
    retained.reverse();
    retained
}

async fn coalesce_hyperliquid_fire_and_forget_request(
    mut request: LiveOrderRequest,
    pending: &mut VecDeque<LiveOrderRequest>,
    order_rx: &mut mpsc::Receiver<LiveOrderRequest>,
    pending_priority: &mut VecDeque<LiveOrderRequest>,
    priority_order_rx: &mut mpsc::Receiver<LiveOrderRequest>,
    window: Duration,
) -> LiveOrderRequest {
    let mut merged_requests = 1usize;
    let mut merged_intents = request.intents.len();

    while pending
        .front()
        .is_some_and(request_contains_hyperliquid_batchable_intents)
    {
        let next = pending
            .pop_front()
            .expect("pending front exists when merging hyperliquid batch window");
        merged_intents += merge_fire_and_forget_request(&mut request, next);
        merged_requests += 1;
    }

    let deadline = tokio::time::Instant::now() + window;
    loop {
        let sleep = tokio::time::sleep_until(deadline);
        tokio::pin!(sleep);
        tokio::select! {
            biased;
            Some(priority_req) = priority_order_rx.recv() => {
                pending_priority.push_back(priority_req);
                break;
            }
            _ = &mut sleep => break,
            maybe_req = order_rx.recv() => {
                let Some(next) = maybe_req else {
                    break;
                };
                if request_contains_hyperliquid_batchable_intents(&next) {
                    merged_intents += merge_fire_and_forget_request(&mut request, next);
                    merged_requests += 1;
                    continue;
                }
                pending.push_back(next);
                break;
            }
        }
    }

    if merged_requests > 1 {
        eprintln!(
            "BATCH_WINDOW_FLUSH venue=hyperliquid window_ms={} merged_requests={} merged_intents={} tick={}",
            window.as_millis(),
            merged_requests,
            merged_intents,
            request.action_batch.tick_index,
        );
    }

    if hyperliquid_mm_batch_same_side_dedup_enabled() {
        let pre_dedup_intents = request.intents.len();
        request.intents = dedup_hyperliquid_mm_same_side_places(request.intents);
        let dropped_intents = pre_dedup_intents.saturating_sub(request.intents.len());
        if dropped_intents > 0 {
            eprintln!(
                "BATCH_WINDOW_DEDUP venue=hyperliquid rule=mm_same_side_places dropped_intents={} tick={}",
                dropped_intents,
                request.action_batch.tick_index,
            );
        }
    }

    request
}

fn normalize_hyperliquid_batch_window_expanded_intents(
    intents: Vec<paraphina::types::OrderIntent>,
) -> Vec<paraphina::types::OrderIntent> {
    let mut passthrough = Vec::with_capacity(intents.len());
    let mut cancel_by_cloid = Vec::new();
    let mut cancel_oid = Vec::new();
    let mut replace_alo = Vec::new();
    let mut place_alo = Vec::new();

    for intent in intents {
        match intent {
            paraphina::types::OrderIntent::Cancel(cancel)
                if cancel.venue_id.eq_ignore_ascii_case("hyperliquid") =>
            {
                if is_hyperliquid_cloid(&cancel.order_id) {
                    cancel_by_cloid.push(paraphina::types::OrderIntent::Cancel(cancel));
                } else {
                    cancel_oid.push(paraphina::types::OrderIntent::Cancel(cancel));
                }
            }
            paraphina::types::OrderIntent::Place(place)
                if place.venue_id.eq_ignore_ascii_case("hyperliquid") && place.post_only =>
            {
                place_alo.push(paraphina::types::OrderIntent::Place(place));
            }
            paraphina::types::OrderIntent::Replace(replace)
                if replace.venue_id.eq_ignore_ascii_case("hyperliquid")
                    && replace.post_only
                    && !replace.reduce_only =>
            {
                replace_alo.push(paraphina::types::OrderIntent::Replace(replace));
            }
            other => passthrough.push(other),
        }
    }

    passthrough.extend(cancel_by_cloid);
    passthrough.extend(cancel_oid);
    passthrough.extend(replace_alo);
    passthrough.extend(place_alo);
    passthrough
}

fn execution_events_from_gateway_result(
    intent: paraphina::types::OrderIntent,
    result: paraphina::live::gateway::LiveResult<paraphina::live::gateway::LiveRestResponse>,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    use paraphina::live::types::{
        CancelAccepted, CancelAllAccepted, CancelAllRejected, CancelRejected, ExecutionEvent,
        OrderAccepted, OrderRejected,
    };

    match intent {
        paraphina::types::OrderIntent::Place(place) => {
            *seq = seq.wrapping_add(1);
            match result {
                Ok(resp) => vec![ExecutionEvent::OrderAccepted(OrderAccepted {
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
                    client_order_id: resp.client_order_id.or(place.client_order_id.clone()),
                    side: place.side,
                    price: place.price,
                    size: place.size,
                    purpose: place.purpose,
                })],
                Err(err) => vec![ExecutionEvent::OrderRejected(OrderRejected {
                    venue_index: place.venue_index,
                    venue_id: place.venue_id.to_string(),
                    seq: *seq,
                    timestamp_ms: now_ms,
                    order_id: place.client_order_id.clone(),
                    client_order_id: place.client_order_id.clone(),
                    purpose: Some(place.purpose),
                    reduce_only: Some(place.reduce_only),
                    reason: err.message.clone(),
                })],
            }
        }
        paraphina::types::OrderIntent::Cancel(cancel) => {
            *seq = seq.wrapping_add(1);
            match result {
                Ok(_resp) => vec![ExecutionEvent::CancelAccepted(CancelAccepted {
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.to_string(),
                    seq: *seq,
                    timestamp_ms: now_ms,
                    order_id: cancel.order_id,
                })],
                Err(err) => vec![ExecutionEvent::CancelRejected(CancelRejected {
                    venue_index: cancel.venue_index,
                    venue_id: cancel.venue_id.to_string(),
                    seq: *seq,
                    timestamp_ms: now_ms,
                    order_id: Some(cancel.order_id),
                    reason: err.message.clone(),
                })],
            }
        }
        paraphina::types::OrderIntent::CancelAll(cancel_all) => {
            *seq = seq.wrapping_add(1);
            let venue_index = cancel_all.venue_index.unwrap_or(0);
            let venue_id = cancel_all
                .venue_id
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "all".to_string());
            match result {
                Ok(_resp) => vec![ExecutionEvent::CancelAllAccepted(CancelAllAccepted {
                    venue_index,
                    venue_id,
                    seq: *seq,
                    timestamp_ms: now_ms,
                    count: 0,
                })],
                Err(err) => vec![ExecutionEvent::CancelAllRejected(CancelAllRejected {
                    venue_index,
                    venue_id,
                    seq: *seq,
                    timestamp_ms: now_ms,
                    reason: err.message.clone(),
                })],
            }
        }
        paraphina::types::OrderIntent::Replace(replace) => match result {
            Ok(resp) => {
                let client_order_id = resp
                    .client_order_id
                    .clone()
                    .or(replace.client_order_id.clone());
                let order_id = resp.order_id.clone().or_else(|| client_order_id.clone());
                let mut events = Vec::with_capacity(2);
                *seq = seq.wrapping_add(1);
                events.push(ExecutionEvent::CancelAccepted(CancelAccepted {
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.to_string(),
                    seq: *seq,
                    timestamp_ms: now_ms,
                    order_id: replace.order_id,
                }));
                *seq = seq.wrapping_add(1);
                events.push(ExecutionEvent::OrderAccepted(OrderAccepted {
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.to_string(),
                    seq: *seq,
                    timestamp_ms: now_ms,
                    order_id: order_id.unwrap_or_else(|| "unknown".to_string()),
                    client_order_id,
                    side: replace.side,
                    price: replace.price,
                    size: replace.size,
                    purpose: replace.purpose,
                }));
                events
            }
            Err(err) => {
                *seq = seq.wrapping_add(1);
                vec![ExecutionEvent::OrderRejected(OrderRejected {
                    venue_index: replace.venue_index,
                    venue_id: replace.venue_id.to_string(),
                    seq: *seq,
                    timestamp_ms: now_ms,
                    order_id: Some(replace.order_id),
                    client_order_id: replace.client_order_id.clone(),
                    purpose: Some(replace.purpose),
                    reduce_only: Some(replace.reduce_only),
                    reason: err.message.clone(),
                })]
            }
        },
    }
}

async fn handle_live_gateway_intent<C: LiveRestClient>(
    gateway: &mut LiveGateway<C>,
    intent: paraphina::types::OrderIntent,
    tick: u64,
    now_ms: paraphina::types::TimestampMs,
    seq: &mut u64,
) -> Vec<paraphina::live::types::ExecutionEvent> {
    match intent {
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
            let result = gateway
                .submit_intent(
                    &paraphina::types::OrderIntent::Replace(replace.clone()),
                    tick,
                    now_ms,
                    TransportHint::Default,
                )
                .await;
            execution_events_from_gateway_result(
                paraphina::types::OrderIntent::Replace(replace),
                result,
                now_ms,
                seq,
            )
        }
    }
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
            TransportHint::Default,
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
                client_order_id: place.client_order_id.clone(),
                purpose: Some(place.purpose),
                reduce_only: Some(place.reduce_only),
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
            TransportHint::Default,
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
            TransportHint::Default,
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

fn coalesce_fire_and_forget_request(
    mut request: LiveOrderRequest,
    pending: &mut VecDeque<LiveOrderRequest>,
    order_rx: &mut mpsc::Receiver<LiveOrderRequest>,
) -> LiveOrderRequest {
    while let Some(next) = pending.pop_front() {
        request = next;
    }
    while let Ok(next) = order_rx.try_recv() {
        request = next;
    }
    request
}

fn take_priority_request(
    pending: &mut VecDeque<LiveOrderRequest>,
    priority_order_rx: &mut mpsc::Receiver<LiveOrderRequest>,
) -> Option<LiveOrderRequest> {
    while let Ok(req) = priority_order_rx.try_recv() {
        pending.push_back(req);
    }
    if let Some(index) = pending
        .iter()
        .position(request_contains_critical_exit_flatten)
    {
        return pending.remove(index);
    }
    pending.pop_front()
}

fn request_contains_critical_exit_flatten(request: &LiveOrderRequest) -> bool {
    request.intents.iter().any(|intent| {
        matches!(
            intent,
            paraphina::types::OrderIntent::Place(place)
                if place.purpose == paraphina::types::OrderPurpose::Exit
                    && place.reduce_only
                    && !place.post_only
                    && place.time_in_force == paraphina::types::TimeInForce::Ioc
        )
    })
}

fn forward_exec_events(
    exec_tx: &mpsc::Sender<paraphina::live::types::ExecutionEvent>,
    events: Vec<paraphina::live::types::ExecutionEvent>,
) {
    for event in events {
        if exec_tx.try_send(event).is_err() {
            eprintln!("paraphina_live | exec_tx full | dropped execution event");
        }
    }
}

fn respond_to_order_request(
    response: ResponseMode,
    events: Vec<paraphina::live::types::ExecutionEvent>,
    exec_tx: &mpsc::Sender<paraphina::live::types::ExecutionEvent>,
) {
    match response {
        ResponseMode::Oneshot(tx) => {
            if let Err(events) = tx.send(events) {
                eprintln!("paraphina_live | order_response_fallback=exec_tx");
                forward_exec_events(exec_tx, events);
            }
        }
        ResponseMode::FireAndForget => {
            forward_exec_events(exec_tx, events);
        }
    }
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

fn register_account_refresh_handler(
    handlers: &mut HashMap<usize, AccountRefreshHandler>,
    connector: &'static str,
    venue_id: String,
    venue_index: usize,
    fetch: AccountRefreshFetch,
) {
    if let Some(previous) = handlers.insert(
        venue_index,
        AccountRefreshHandler {
            connector,
            venue_id: venue_id.clone(),
            venue_index,
            fetch,
        },
    ) {
        eprintln!(
            "paraphina_live | account_refresh_handler_replaced venue_index={} previous_connector={} previous_venue={} connector={} venue={}",
            venue_index, previous.connector, previous.venue_id, connector, venue_id
        );
    }
}

fn spawn_account_refresh_router(
    mut request_rx: mpsc::Receiver<LiveAccountRequest>,
    handlers: HashMap<usize, AccountRefreshHandler>,
) {
    tokio::spawn(async move {
        while let Some(request) = request_rx.recv().await {
            let requested_venue_index = request.venue_index.unwrap_or(usize::MAX);
            let Some(handler) = handlers.get(&requested_venue_index).cloned() else {
                eprintln!(
                    "paraphina_live | account_refresh_unsupported requested_venue_index={}",
                    requested_venue_index
                );
                continue;
            };
            let request_age_ms = now_ms().saturating_sub(request.now_ms);
            match (handler.fetch)().await {
                Ok(snapshot) => {
                    let timestamp_ms = snapshot.timestamp_ms;
                    if request.response.send(snapshot).is_err() {
                        eprintln!(
                            "paraphina_live | account_refresh_response_dropped connector={} venue={} venue_index={} snapshot_ts_ms={}",
                            handler.connector, handler.venue_id, handler.venue_index, timestamp_ms
                        );
                    }
                }
                Err(err) => {
                    eprintln!(
                        "paraphina_live | account_refresh_failed connector={} venue={} venue_index={} request_age_ms={} err={}",
                        handler.connector, handler.venue_id, handler.venue_index, request_age_ms, err.message
                    );
                }
            }
        }
    });
}

fn write_summary(
    out_dir: &std::path::Path,
    cfg: &Config,
    trade_mode: TradeMode,
    connector: &str,
    summary: &paraphina::live::LiveRunSummary,
    stats: Arc<LiveTelemetryStats>,
) {
    use std::sync::atomic::Ordering as AO;
    let ticks_total = stats.ticks_total.load(AO::Relaxed);
    let fv_available_ticks = stats.fv_available_ticks.load(AO::Relaxed);
    let venue_staleness_events = stats.venue_staleness_events.load(AO::Relaxed);
    let venue_disabled_events = stats.venue_disabled_events.load(AO::Relaxed);
    let kill_events = stats.kill_events.load(AO::Relaxed);
    let maps = stats.purpose_maps.lock().ok().map(|g| g.clone());

    let fv_rate = if ticks_total > 0 {
        fv_available_ticks as f64 / ticks_total as f64
    } else {
        0.0
    };
    let (place_map, cancel_map, replace_map) = match maps {
        Some(m) => (
            m.would_place_by_purpose,
            m.would_cancel_by_purpose,
            m.would_replace_by_purpose,
        ),
        None => (Default::default(), Default::default(), Default::default()),
    };
    let payload = serde_json::json!({
        "trade_mode": trade_mode.as_str(),
        "execution_mode": trade_mode.as_str(),
        "connector": connector,
        "venues": cfg.venues.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(),
        "ticks_run": summary.ticks_run,
        "run_duration_ms": summary.ticks_run as i64 * cfg.main_loop_interval_ms,
        "would_place_by_purpose": place_map,
        "would_cancel_by_purpose": cancel_map,
        "would_replace_by_purpose": replace_map,
        "fv_available_rate": fv_rate,
        "venue_staleness_events": venue_staleness_events,
        "venue_disabled_events": venue_disabled_events,
        "kill_events": kill_events,
    });
    let path = out_dir.join("summary.json");
    if let Ok(text) = serde_json::to_string_pretty(&payload) {
        let _ = std::fs::write(path, text);
    }
}

#[cfg(test)]
mod tests {
    use super::register_account_refresh_handler;
    use super::{
        coalesce_fire_and_forget_request, coalesce_hyperliquid_fire_and_forget_request,
        connector_market_channel_cap, dedup_hyperliquid_mm_same_side_places, market_channel_cap,
        market_ingest_channel_cap, normalize_hyperliquid_batch_window_expanded_intents,
        paradex_private_order_truth_enabled, parse_aster_backstop_account_poll_ms,
        parse_paradex_backstop_order_poll_ms, respond_to_order_request,
        should_fail_on_unexpected_live_loop_exit, spawn_account_refresh_router,
        take_priority_request, LiveAccountRequest, LiveOrderRequest, LiveRunMode, ResponseMode,
    };
    use super::{
        connector_has_passive_fill_stream, connector_has_passive_fill_visibility,
        connector_support, ConnectorArg, ConnectorSupport,
    };
    use paraphina::actions::ActionBatch;
    use paraphina::live::gateway::TransportHint;
    use paraphina::live::types::{
        AccountSnapshot, BalanceSnapshot, ExecutionEvent, LiquidationSnapshot, MarginSnapshot,
        OrderAccepted, PositionSnapshot,
    };
    use paraphina::live::venues::ROADMAP_B_VENUES;
    use paraphina::types::{
        CancelAllOrderIntent, CancelOrderIntent, OrderIntent, OrderPurpose,
        Phase51ForwardRefreshTargetKey, PlaceOrderIntent, ReplaceOrderIntent, Side, TimeInForce,
    };
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;
    use tokio::sync::{mpsc, oneshot};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn phase51_target_key(label: &str) -> Phase51ForwardRefreshTargetKey {
        Phase51ForwardRefreshTargetKey {
            canonical_group_id: format!("canonical-{label}"),
            order_key: format!("order-{label}"),
        }
    }

    struct EnvVarGuard {
        key: &'static str,
        prior: Option<String>,
    }

    impl EnvVarGuard {
        fn new(key: &'static str) -> Self {
            Self {
                key,
                prior: std::env::var(key).ok(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(value) = &self.prior {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    fn test_account_snapshot(venue_index: usize, venue_id: &str, size: f64) -> AccountSnapshot {
        AccountSnapshot {
            venue_index,
            venue_id: venue_id.to_string(),
            seq: 7,
            timestamp_ms: 1_234,
            positions: vec![PositionSnapshot {
                symbol: "ETH-USD".to_string(),
                size,
                entry_price: 2_000.0,
            }],
            balances: vec![BalanceSnapshot {
                asset: "USD".to_string(),
                total: 100.0,
                available: 90.0,
            }],
            funding_8h: None,
            margin: MarginSnapshot {
                balance_usd: 100.0,
                used_usd: 10.0,
                available_usd: 90.0,
            },
            liquidation: LiquidationSnapshot {
                price_liq: None,
                dist_liq_sigma: None,
            },
        }
    }

    #[tokio::test]
    async fn account_refresh_router_routes_by_venue_index() {
        let (request_tx, request_rx) = mpsc::channel(4);
        let mut handlers = HashMap::new();
        register_account_refresh_handler(
            &mut handlers,
            "extended",
            "extended".to_string(),
            2,
            Arc::new(|| Box::pin(async { Ok(test_account_snapshot(2, "extended", 0.01)) })),
        );
        spawn_account_refresh_router(request_rx, handlers);

        let (response_tx, response_rx) = oneshot::channel();
        request_tx
            .send(LiveAccountRequest {
                venue_index: Some(2),
                now_ms: 1,
                response: response_tx,
            })
            .await
            .expect("send account refresh request");

        let snapshot = tokio::time::timeout(Duration::from_secs(1), response_rx)
            .await
            .expect("account refresh response timed out")
            .expect("account refresh response dropped");
        assert_eq!(snapshot.venue_index, 2);
        assert_eq!(snapshot.venue_id, "extended");
        assert_eq!(snapshot.positions[0].size, 0.01);
    }

    #[test]
    fn roadmap_b_registry_is_complete() {
        assert_eq!(ROADMAP_B_VENUES.len(), 5);
        let selectable = ConnectorArg::roadmap_b_selectable_venues();
        assert_eq!(selectable, ROADMAP_B_VENUES.to_vec());
    }

    #[test]
    fn explicit_lighter_only_selection_clamps_min_healthy_for_kf() {
        let mut cfg = paraphina::config::Config::default();
        cfg.book.min_healthy_for_kf = 2;

        super::apply_explicit_connector_selection_to_config(&mut cfg, &[ConnectorArg::Lighter]);

        assert_eq!(cfg.venues.len(), 1);
        assert_eq!(cfg.venues[0].id, "lighter");
        assert_eq!(cfg.book.min_healthy_for_kf, 1);
    }

    #[test]
    fn explicit_multi_venue_selection_preserves_satisfied_min_healthy_for_kf() {
        let mut cfg = paraphina::config::Config::default();
        cfg.book.min_healthy_for_kf = 2;

        super::apply_explicit_connector_selection_to_config(
            &mut cfg,
            &[ConnectorArg::Hyperliquid, ConnectorArg::Lighter],
        );

        assert_eq!(
            cfg.venues
                .iter()
                .map(|venue| venue.id.as_str())
                .collect::<Vec<_>>(),
            vec!["hyperliquid", "lighter"]
        );
        assert_eq!(cfg.book.min_healthy_for_kf, 2);
    }

    #[test]
    fn explicit_selection_does_not_raise_existing_min_healthy_for_kf() {
        let mut cfg = paraphina::config::Config::default();
        cfg.book.min_healthy_for_kf = 1;

        super::apply_explicit_connector_selection_to_config(
            &mut cfg,
            &[ConnectorArg::Hyperliquid, ConnectorArg::Lighter],
        );

        assert_eq!(cfg.book.min_healthy_for_kf, 1);
    }

    #[test]
    fn canary_profile_can_override_book_max_mid_jump_pct() {
        let mut cfg = paraphina::config::Config::default();
        cfg.book.max_mid_jump_pct = 0.02;
        let canary = super::CanaryConfig {
            venue: None,
            limits: None,
            rate_limit: None,
            enforcement: None,
            kill: None,
            book: Some(super::CanaryBookConfig {
                max_mid_jump_pct: Some(0.03),
            }),
        };

        let settings = super::apply_canary_config(&mut cfg, &canary);

        assert_eq!(settings.max_mid_jump_pct, Some(0.03));
        assert_eq!(cfg.book.max_mid_jump_pct, 0.03);
    }

    #[test]
    fn aster_backstop_account_poll_ms_prefers_specific_override() {
        const KEY: &str = "PARAPHINA_ASTER_BACKSTOP_ACCOUNT_POLL_MS";
        let prior = std::env::var(KEY).ok();
        std::env::remove_var(KEY);
        assert_eq!(parse_aster_backstop_account_poll_ms(), 15_000);
        std::env::set_var(KEY, "21000");
        assert_eq!(parse_aster_backstop_account_poll_ms(), 21_000);
        if let Some(value) = prior {
            std::env::set_var(KEY, value);
        } else {
            std::env::remove_var(KEY);
        }
    }

    #[test]
    fn paradex_private_order_truth_enabled_reads_boolean_env() {
        const KEY: &str = "PARAPHINA_PARADEX_PRIVATE_ORDER_TRUTH_ENABLED";
        let prior = std::env::var(KEY).ok();
        std::env::remove_var(KEY);
        assert!(!paradex_private_order_truth_enabled());
        std::env::set_var(KEY, "1");
        assert!(paradex_private_order_truth_enabled());
        std::env::set_var(KEY, "true");
        assert!(paradex_private_order_truth_enabled());
        if let Some(value) = prior {
            std::env::set_var(KEY, value);
        } else {
            std::env::remove_var(KEY);
        }
    }

    #[test]
    fn market_frontier_channel_caps_read_env_overrides() {
        let _guard = env_lock().lock().expect("env mutex");
        std::env::set_var("PARAPHINA_MARKET_INGEST_CHANNEL_CAP", "4096");
        std::env::set_var("PARAPHINA_MARKET_CHANNEL_CAP", "4096");
        std::env::set_var("PARAPHINA_CONNECTOR_MARKET_CHANNEL_CAP", "2048");
        assert_eq!(market_ingest_channel_cap(), 4096);
        assert_eq!(market_channel_cap(), 4096);
        assert_eq!(connector_market_channel_cap(), 2048);
        std::env::remove_var("PARAPHINA_MARKET_INGEST_CHANNEL_CAP");
        std::env::remove_var("PARAPHINA_MARKET_CHANNEL_CAP");
        std::env::remove_var("PARAPHINA_CONNECTOR_MARKET_CHANNEL_CAP");
    }

    #[test]
    fn paradex_backstop_order_poll_ms_prefers_specific_override() {
        const KEY: &str = "PARAPHINA_PARADEX_ORDER_BACKSTOP_POLL_MS";
        let prior = std::env::var(KEY).ok();
        std::env::remove_var(KEY);
        assert_eq!(parse_paradex_backstop_order_poll_ms(), 15_000);
        std::env::set_var(KEY, "21000");
        assert_eq!(parse_paradex_backstop_order_poll_ms(), 21_000);
        if let Some(value) = prior {
            std::env::set_var(KEY, value);
        } else {
            std::env::remove_var(KEY);
        }
    }

    #[test]
    fn expand_live_gateway_intents_keeps_native_hyperliquid_replace() {
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 2,
            venue_id: "hyperliquid".into(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
            client_order_id: Some("new-coid".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 1);
        match &expanded[0] {
            OrderIntent::Replace(replace) => {
                assert_eq!(replace.venue_id.as_ref(), "hyperliquid");
                assert_eq!(replace.client_order_id.as_deref(), Some("new-coid"));
                assert!(replace.post_only);
            }
            other => panic!("expected replace, got {other:?}"),
        }
    }

    #[test]
    fn expand_live_gateway_intents_keeps_native_hyperliquid_replace_for_numeric_oid() {
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 2,
            venue_id: "hyperliquid".into(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "123456789".to_string(),
            client_order_id: Some("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 1);
        assert!(matches!(expanded[0], OrderIntent::Replace(_)));
    }

    #[test]
    fn expand_live_gateway_intents_still_splits_non_native_replace() {
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 1,
            venue_id: "lighter".into(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "old-order-id".to_string(),
            client_order_id: Some("co_lighter".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 2);
        assert!(matches!(expanded[0], OrderIntent::Cancel(_)));
        assert!(matches!(expanded[1], OrderIntent::Place(_)));
    }

    #[test]
    fn expand_live_gateway_intents_preserves_explicit_phase51_target_key_for_split_replace() {
        let target_key = phase51_target_key("split-replace");
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 3,
            venue_id: "aster".into(),
            side: Side::Sell,
            price: 2_105.0,
            size: 0.04,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "previous-handle".to_string(),
            client_order_id: Some("replacement-client-handle".to_string()),
            phase51_target_key: Some(target_key.clone()),
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 2);
        assert!(matches!(expanded[0], OrderIntent::Cancel(_)));
        let preserved = match &expanded[1] {
            OrderIntent::Place(place) => match place.phase51_target_key.as_ref() {
                Some(key) => {
                    key.canonical_group_id == target_key.canonical_group_id
                        && key.order_key == target_key.order_key
                }
                None => false,
            },
            _ => false,
        };
        assert!(preserved);
    }

    #[test]
    fn expand_live_gateway_intents_keeps_absent_phase51_target_key_none_for_split_replace() {
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 3,
            venue_id: "aster".into(),
            side: Side::Buy,
            price: 2_106.0,
            size: 0.03,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "previous-handle".to_string(),
            client_order_id: Some("replacement-client-handle".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 2);
        match &expanded[1] {
            OrderIntent::Place(place) => assert!(place.phase51_target_key.is_none()),
            _ => panic!("expected place intent"),
        }
    }

    #[test]
    fn expand_live_gateway_intents_does_not_derive_phase51_target_key_for_generated_handles() {
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 3,
            venue_id: "aster".into(),
            side: Side::Sell,
            price: 2_107.0,
            size: 0.02,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "generated-looking-previous-handle".to_string(),
            client_order_id: Some("generated-looking-replacement-handle".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 2);
        match &expanded[1] {
            OrderIntent::Place(place) => assert!(place.phase51_target_key.is_none()),
            _ => panic!("expected place intent"),
        }
    }

    #[test]
    fn expand_live_gateway_intents_keeps_native_lighter_replace_for_numeric_id() {
        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 1,
            venue_id: "lighter".into(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "55".to_string(),
            client_order_id: Some("77".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 1);
        match &expanded[0] {
            OrderIntent::Replace(replace) => {
                assert_eq!(replace.venue_id.as_ref(), "lighter");
                assert_eq!(replace.order_id, "55");
                assert_eq!(replace.client_order_id.as_deref(), Some("77"));
            }
            other => panic!("expected replace, got {other:?}"),
        }
    }

    #[test]
    fn expand_live_gateway_intents_keeps_gated_extended_replace_for_external_id() {
        const KEY: &str = "PARAPHINA_EXTENDED_NATIVE_REPLACE_ENABLED";
        let _guard = env_lock().lock().expect("env mutex");
        let _env_guard = EnvVarGuard::new(KEY);
        std::env::set_var(KEY, "1");

        let intents = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 4,
            venue_id: "extended".into(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "d0_mm_v4_buy".to_string(),
            client_order_id: Some("d1_mm_v4_buy".to_string()),
            phase51_target_key: None,
        })];
        let expanded = super::expand_live_gateway_intents(intents);
        assert_eq!(expanded.len(), 1);
        match &expanded[0] {
            OrderIntent::Replace(replace) => {
                assert_eq!(replace.venue_id.as_ref(), "extended");
                assert_eq!(replace.order_id, "d0_mm_v4_buy");
                assert_eq!(replace.client_order_id.as_deref(), Some("d1_mm_v4_buy"));
            }
            other => panic!("expected replace, got {other:?}"),
        }
    }

    #[test]
    fn expand_live_gateway_intents_splits_extended_replace_when_disabled_or_numeric() {
        const KEY: &str = "PARAPHINA_EXTENDED_NATIVE_REPLACE_ENABLED";
        let _guard = env_lock().lock().expect("env mutex");
        let _env_guard = EnvVarGuard::new(KEY);
        std::env::remove_var(KEY);

        let disabled = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 4,
            venue_id: "extended".into(),
            side: Side::Buy,
            price: 2_100.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "d0_mm_v4_buy".to_string(),
            client_order_id: Some("d1_mm_v4_buy".to_string()),
            phase51_target_key: None,
        })];
        let expanded_disabled = super::expand_live_gateway_intents(disabled);
        assert_eq!(expanded_disabled.len(), 2);
        assert!(matches!(expanded_disabled[0], OrderIntent::Cancel(_)));
        assert!(matches!(expanded_disabled[1], OrderIntent::Place(_)));

        std::env::set_var(KEY, "1");
        let numeric = vec![OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 4,
            venue_id: "extended".into(),
            side: Side::Sell,
            price: 2_101.0,
            size: 0.05,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "1784963886257016832".to_string(),
            client_order_id: Some("d1_mm_v4_sell".to_string()),
            phase51_target_key: None,
        })];
        let expanded_numeric = super::expand_live_gateway_intents(numeric);
        assert_eq!(expanded_numeric.len(), 2);
        assert!(matches!(expanded_numeric[0], OrderIntent::Cancel(_)));
        assert!(matches!(expanded_numeric[1], OrderIntent::Place(_)));
    }

    #[test]
    fn replace_gateway_result_emits_cancel_then_order_accept() {
        let intent = OrderIntent::Replace(ReplaceOrderIntent {
            venue_index: 4,
            venue_id: "paradex".into(),
            side: Side::Sell,
            price: 101.0,
            size: 0.01,
            purpose: OrderPurpose::Mm,
            time_in_force: TimeInForce::Gtc,
            post_only: true,
            reduce_only: false,
            order_id: "pdx_old".to_string(),
            client_order_id: Some("co_pdx_new".to_string()),
            phase51_target_key: None,
        });
        let mut seq = 10_u64;
        let events = super::execution_events_from_gateway_result(
            intent,
            Ok(paraphina::live::gateway::LiveRestResponse {
                order_id: Some("pdx_new".to_string()),
                client_order_id: Some("co_pdx_new".to_string()),
            }),
            1_000,
            &mut seq,
        );
        assert_eq!(events.len(), 2);
        match &events[0] {
            ExecutionEvent::CancelAccepted(cancel) => assert_eq!(cancel.order_id, "pdx_old"),
            other => panic!("expected cancel accepted, got {other:?}"),
        }
        match &events[1] {
            ExecutionEvent::OrderAccepted(ack) => {
                assert_eq!(ack.order_id, "pdx_new");
                assert_eq!(ack.client_order_id.as_deref(), Some("co_pdx_new"));
            }
            other => panic!("expected order accepted, got {other:?}"),
        }
    }

    #[test]
    fn cancel_batch_error_maps_to_cancel_reject_not_place_reject() {
        let intent = OrderIntent::Cancel(paraphina::types::CancelOrderIntent {
            venue_index: 0,
            venue_id: "hyperliquid".into(),
            order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
        });
        let mut seq = 0_u64;
        let events = super::execution_events_from_gateway_result(
            intent,
            Err(paraphina::live::gateway::LiveGatewayError::fatal(
                "Hyperliquid cancel_batch failed: 422 Unprocessable Entity",
            )),
            1_000,
            &mut seq,
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            ExecutionEvent::CancelRejected(reject) => {
                assert_eq!(reject.venue_id, "hyperliquid");
                assert_eq!(
                    reject.reason,
                    "Hyperliquid cancel_batch failed: 422 Unprocessable Entity"
                );
            }
            other => panic!("expected CancelRejected, got {other:?}"),
        }
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

    #[test]
    fn passive_fill_stream_matrix_matches_current_connectors() {
        assert!(connector_has_passive_fill_stream(ConnectorArg::Hyperliquid));
        assert!(connector_has_passive_fill_stream(ConnectorArg::Lighter));
        assert!(!connector_has_passive_fill_stream(ConnectorArg::Extended));
        assert!(!connector_has_passive_fill_stream(ConnectorArg::Aster));
        assert!(!connector_has_passive_fill_stream(ConnectorArg::Paradex));
    }

    #[test]
    fn passive_fill_visibility_matrix_matches_current_connectors() {
        assert!(connector_has_passive_fill_visibility(
            ConnectorArg::Hyperliquid
        ));
        assert!(connector_has_passive_fill_visibility(ConnectorArg::Lighter));
        assert!(connector_has_passive_fill_visibility(
            ConnectorArg::Extended
        ));
        assert!(connector_has_passive_fill_visibility(ConnectorArg::Aster));
        assert!(connector_has_passive_fill_visibility(ConnectorArg::Paradex));
    }

    #[tokio::test]
    async fn fire_and_forget_requests_coalesce_to_latest_batch() {
        let (order_tx, mut order_rx) = mpsc::channel(8);
        order_tx
            .send(LiveOrderRequest {
                intents: Vec::new(),
                action_batch: ActionBatch::new(0, 1, "test"),
                now_ms: 1,
                transport_hint: TransportHint::Default,
                response: ResponseMode::FireAndForget,
            })
            .await
            .expect("send first fire-and-forget");
        order_tx
            .send(LiveOrderRequest {
                intents: Vec::new(),
                action_batch: ActionBatch::new(0, 2, "test"),
                now_ms: 2,
                transport_hint: TransportHint::Default,
                response: ResponseMode::FireAndForget,
            })
            .await
            .expect("send second fire-and-forget");

        let first_req = order_rx.recv().await.expect("first request");
        let mut pending = VecDeque::from([LiveOrderRequest {
            intents: Vec::new(),
            action_batch: ActionBatch::new(0, 3, "test"),
            now_ms: 3,
            transport_hint: TransportHint::Default,
            response: ResponseMode::FireAndForget,
        }]);
        let coalesced = coalesce_fire_and_forget_request(first_req, &mut pending, &mut order_rx);

        assert!(matches!(coalesced.response, ResponseMode::FireAndForget));
        assert_eq!(coalesced.action_batch.tick_index, 2);
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn hyperliquid_fire_and_forget_requests_coalesce_within_batch_window() {
        let (order_tx, mut order_rx) = mpsc::channel(8);
        let (_priority_order_tx, mut priority_order_rx) = mpsc::channel(8);
        let mut pending_priority = VecDeque::new();
        let venue_id = Arc::<str>::from("hyperliquid");

        order_tx
            .send(LiveOrderRequest {
                intents: vec![OrderIntent::Place(PlaceOrderIntent {
                    venue_index: 0,
                    venue_id: venue_id.clone(),
                    side: Side::Buy,
                    price: 2_100.0,
                    size: 0.02,
                    purpose: OrderPurpose::Mm,
                    time_in_force: TimeInForce::Gtc,
                    post_only: true,
                    reduce_only: false,
                    client_order_id: Some("co_queued".to_string()),
                    phase51_target_key: None,
                })],
                action_batch: ActionBatch::new(0, 3, "test"),
                now_ms: 3,
                transport_hint: TransportHint::Default,
                response: ResponseMode::FireAndForget,
            })
            .await
            .expect("send queued hyperliquid place");

        let first_req = LiveOrderRequest {
            intents: vec![OrderIntent::Cancel(CancelOrderIntent {
                venue_index: 0,
                venue_id: venue_id.clone(),
                order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
            })],
            action_batch: ActionBatch::new(0, 1, "test"),
            now_ms: 1,
            transport_hint: TransportHint::Default,
            response: ResponseMode::FireAndForget,
        };
        let mut pending = VecDeque::from([LiveOrderRequest {
            intents: vec![OrderIntent::Place(PlaceOrderIntent {
                venue_index: 0,
                venue_id,
                side: Side::Sell,
                price: 2_101.0,
                size: 0.02,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("co_pending".to_string()),
                phase51_target_key: None,
            })],
            action_batch: ActionBatch::new(0, 2, "test"),
            now_ms: 2,
            transport_hint: TransportHint::Default,
            response: ResponseMode::FireAndForget,
        }]);

        let coalesced = coalesce_hyperliquid_fire_and_forget_request(
            first_req,
            &mut pending,
            &mut order_rx,
            &mut pending_priority,
            &mut priority_order_rx,
            Duration::from_millis(1),
        )
        .await;

        assert_eq!(coalesced.intents.len(), 3);
        assert_eq!(coalesced.action_batch.tick_index, 3);
        assert!(matches!(coalesced.response, ResponseMode::FireAndForget));
        assert!(pending.is_empty());
        assert!(pending_priority.is_empty());
    }

    #[test]
    fn hyperliquid_same_side_mm_place_dedup_keeps_latest_per_side() {
        let hyperliquid_id = Arc::<str>::from("hyperliquid");
        let lighter_id = Arc::<str>::from("lighter");
        let deduped = dedup_hyperliquid_mm_same_side_places(vec![
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 0,
                venue_id: hyperliquid_id.clone(),
                side: Side::Buy,
                price: 2_100.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("hl-buy-old".to_string()),
                phase51_target_key: None,
            }),
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 0,
                venue_id: hyperliquid_id.clone(),
                side: Side::Sell,
                price: 2_101.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("hl-sell".to_string()),
                phase51_target_key: None,
            }),
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 0,
                venue_id: hyperliquid_id.clone(),
                side: Side::Buy,
                price: 2_102.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("hl-buy-latest".to_string()),
                phase51_target_key: None,
            }),
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 0,
                venue_id: hyperliquid_id,
                side: Side::Buy,
                price: 2_099.0,
                size: 0.02,
                purpose: OrderPurpose::Exit,
                time_in_force: TimeInForce::Ioc,
                post_only: false,
                reduce_only: true,
                client_order_id: Some("hl-exit".to_string()),
                phase51_target_key: None,
            }),
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 1,
                venue_id: lighter_id,
                side: Side::Buy,
                price: 2_100.0,
                size: 0.01,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("lighter-buy".to_string()),
                phase51_target_key: None,
            }),
        ]);

        let client_ids: Vec<Option<&str>> = deduped
            .iter()
            .map(|intent| match intent {
                OrderIntent::Place(place) => place.client_order_id.as_deref(),
                _ => None,
            })
            .collect();
        assert_eq!(
            client_ids,
            vec![
                Some("hl-sell"),
                Some("hl-buy-latest"),
                Some("hl-exit"),
                Some("lighter-buy")
            ]
        );
    }

    #[test]
    fn normalize_hyperliquid_expanded_intents_groups_cancels_before_alo_places() {
        let normalized = normalize_hyperliquid_batch_window_expanded_intents(vec![
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 1,
                venue_id: Arc::<str>::from("lighter"),
                side: Side::Buy,
                price: 100.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("lighter-1".to_string()),
                phase51_target_key: None,
            }),
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 0,
                venue_id: Arc::<str>::from("hyperliquid"),
                side: Side::Buy,
                price: 2_100.0,
                size: 0.02,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("hl-place".to_string()),
                phase51_target_key: None,
            }),
            OrderIntent::Cancel(CancelOrderIntent {
                venue_index: 0,
                venue_id: Arc::<str>::from("hyperliquid"),
                order_id: "0x1234567890abcdef1234567890abcdef".to_string(),
            }),
            OrderIntent::Cancel(CancelOrderIntent {
                venue_index: 0,
                venue_id: Arc::<str>::from("hyperliquid"),
                order_id: "12345".to_string(),
            }),
            OrderIntent::Place(PlaceOrderIntent {
                venue_index: 2,
                venue_id: Arc::<str>::from("paradex"),
                side: Side::Sell,
                price: 100.5,
                size: 1.0,
                purpose: OrderPurpose::Mm,
                time_in_force: TimeInForce::Gtc,
                post_only: true,
                reduce_only: false,
                client_order_id: Some("paradex-1".to_string()),
                phase51_target_key: None,
            }),
        ]);

        assert_eq!(normalized.len(), 5);
        match &normalized[0] {
            OrderIntent::Place(place) => assert_eq!(place.venue_id.as_ref(), "lighter"),
            other => panic!("expected lighter passthrough first, got {other:?}"),
        }
        match &normalized[1] {
            OrderIntent::Place(place) => assert_eq!(place.venue_id.as_ref(), "paradex"),
            other => panic!("expected paradex passthrough second, got {other:?}"),
        }
        match &normalized[2] {
            OrderIntent::Cancel(cancel) => {
                assert_eq!(cancel.venue_id.as_ref(), "hyperliquid");
                assert!(cancel.order_id.starts_with("0x"));
            }
            other => panic!("expected hyperliquid cloid cancel third, got {other:?}"),
        }
        match &normalized[3] {
            OrderIntent::Cancel(cancel) => {
                assert_eq!(cancel.venue_id.as_ref(), "hyperliquid");
                assert_eq!(cancel.order_id, "12345");
            }
            other => panic!("expected hyperliquid oid cancel fourth, got {other:?}"),
        }
        match &normalized[4] {
            OrderIntent::Place(place) => {
                assert_eq!(place.venue_id.as_ref(), "hyperliquid");
                assert_eq!(place.client_order_id.as_deref(), Some("hl-place"));
            }
            other => panic!("expected hyperliquid place last, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn priority_requests_surface_ahead_of_fire_and_forget_batches() {
        let (priority_order_tx, mut priority_order_rx) = mpsc::channel(8);
        let (response_tx, response_rx) = oneshot::channel::<Vec<ExecutionEvent>>();
        let mut pending = VecDeque::from([LiveOrderRequest {
            intents: Vec::new(),
            action_batch: ActionBatch::new(0, 11, "test"),
            now_ms: 11,
            transport_hint: TransportHint::Default,
            response: ResponseMode::Oneshot(response_tx),
        }]);

        priority_order_tx
            .send(LiveOrderRequest {
                intents: Vec::new(),
                action_batch: ActionBatch::new(0, 12, "test"),
                now_ms: 12,
                transport_hint: TransportHint::Default,
                response: ResponseMode::Oneshot(oneshot::channel::<Vec<ExecutionEvent>>().0),
            })
            .await
            .expect("send priority request");

        let blocking = take_priority_request(&mut pending, &mut priority_order_rx)
            .expect("priority request should be surfaced");
        assert!(matches!(blocking.response, ResponseMode::Oneshot(_)));
        assert_eq!(blocking.action_batch.tick_index, 11);
        let next = take_priority_request(&mut pending, &mut priority_order_rx)
            .expect("queued priority request should be surfaced next");
        assert!(matches!(next.response, ResponseMode::Oneshot(_)));
        assert_eq!(next.action_batch.tick_index, 12);
        assert!(pending.is_empty());
        drop(response_rx);
    }

    #[tokio::test]
    async fn take_priority_request_promotes_critical_exit_flatten_over_backlog() {
        let (priority_order_tx, mut priority_order_rx) = mpsc::channel(8);
        let mut pending = VecDeque::from([
            LiveOrderRequest {
                intents: vec![OrderIntent::CancelAll(CancelAllOrderIntent {
                    venue_index: Some(1),
                    venue_id: Some(Arc::<str>::from("lighter")),
                })],
                action_batch: ActionBatch::new(0, 31, "test"),
                now_ms: 31,
                transport_hint: TransportHint::Default,
                response: ResponseMode::FireAndForget,
            },
            LiveOrderRequest {
                intents: vec![OrderIntent::Place(PlaceOrderIntent {
                    venue_index: 1,
                    venue_id: Arc::<str>::from("lighter"),
                    side: Side::Buy,
                    price: 2_100.0,
                    size: 0.01,
                    purpose: OrderPurpose::Hedge,
                    time_in_force: TimeInForce::Ioc,
                    post_only: false,
                    reduce_only: true,
                    client_order_id: Some("hedge-flatten".to_string()),
                    phase51_target_key: None,
                })],
                action_batch: ActionBatch::new(0, 32, "test"),
                now_ms: 32,
                transport_hint: TransportHint::Default,
                response: ResponseMode::FireAndForget,
            },
        ]);

        priority_order_tx
            .send(LiveOrderRequest {
                intents: vec![OrderIntent::Place(PlaceOrderIntent {
                    venue_index: 1,
                    venue_id: Arc::<str>::from("lighter"),
                    side: Side::Buy,
                    price: 2_100.0,
                    size: 0.04,
                    purpose: OrderPurpose::Exit,
                    time_in_force: TimeInForce::Ioc,
                    post_only: false,
                    reduce_only: true,
                    client_order_id: Some("critical-exit-flatten".to_string()),
                    phase51_target_key: None,
                })],
                action_batch: ActionBatch::new(0, 33, "test"),
                now_ms: 33,
                transport_hint: TransportHint::Default,
                response: ResponseMode::FireAndForget,
            })
            .await
            .expect("send critical exit flatten");

        let critical = take_priority_request(&mut pending, &mut priority_order_rx)
            .expect("critical exit flatten should be promoted");
        assert_eq!(critical.action_batch.tick_index, 33);
        match &critical.intents[0] {
            OrderIntent::Place(place) => {
                assert_eq!(place.purpose, OrderPurpose::Exit);
                assert!(place.reduce_only);
                assert_eq!(place.time_in_force, TimeInForce::Ioc);
            }
            other => panic!("expected critical exit place, got {other:?}"),
        }

        let oldest = take_priority_request(&mut pending, &mut priority_order_rx)
            .expect("older non-critical request remains queued");
        assert_eq!(oldest.action_batch.tick_index, 31);
    }

    #[tokio::test]
    async fn take_priority_request_preserves_request_transport_hint() {
        let (_priority_order_tx, mut priority_order_rx) = mpsc::channel(1);
        let mut pending = VecDeque::from([LiveOrderRequest {
            intents: Vec::new(),
            action_batch: ActionBatch::new(0, 21, "test"),
            now_ms: 21,
            transport_hint: TransportHint::HyperliquidSyncControl,
            response: ResponseMode::FireAndForget,
        }]);

        let req =
            take_priority_request(&mut pending, &mut priority_order_rx).expect("priority request");

        assert_eq!(req.action_batch.tick_index, 21);
        assert_eq!(req.transport_hint, TransportHint::HyperliquidSyncControl);
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn failed_oneshot_response_falls_back_to_exec_tx() {
        let (exec_tx, mut exec_rx) = mpsc::channel(4);
        let (response_tx, response_rx) = oneshot::channel::<Vec<ExecutionEvent>>();
        drop(response_rx);

        respond_to_order_request(
            ResponseMode::Oneshot(response_tx),
            vec![ExecutionEvent::OrderAccepted(OrderAccepted {
                venue_index: 0,
                venue_id: "aster".to_string(),
                seq: 1,
                timestamp_ms: 1_000,
                order_id: "oid-1".to_string(),
                client_order_id: Some("co_1".to_string()),
                side: Side::Buy,
                price: 100.0,
                size: 1.0,
                purpose: OrderPurpose::Mm,
            })],
            &exec_tx,
        );

        let event = exec_rx.recv().await.expect("fallback exec event");
        assert!(matches!(event, ExecutionEvent::OrderAccepted(_)));
    }

    #[test]
    fn unexpected_unbounded_realtime_loop_exit_fails_closed() {
        assert!(should_fail_on_unexpected_live_loop_exit(
            LiveRunMode::Realtime {
                interval_ms: 250,
                max_ticks: None,
            }
        ));
        assert!(!should_fail_on_unexpected_live_loop_exit(
            LiveRunMode::Realtime {
                interval_ms: 250,
                max_ticks: Some(10),
            }
        ));
        assert!(!should_fail_on_unexpected_live_loop_exit(
            LiveRunMode::Step {
                start_ms: 0,
                step_ms: 250,
                ticks: 10,
            }
        ));
    }
}
