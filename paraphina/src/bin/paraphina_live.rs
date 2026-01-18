//! Live trading skeleton binary (feature-gated).
//!
//! This binary wires the live cache, event model, and strategy loop together
//! without any external network connectors.

use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::path::PathBuf;
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
use paraphina::live::paper_adapter::{PaperExecutionAdapter, PaperFillMode, PaperMarketUpdate};
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

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
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
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("canary_profile_read_error path={} err={}", path.display(), err))?;
    toml::from_str::<CanaryConfig>(&raw)
        .map_err(|err| format!("canary_profile_parse_error path={} err={}", path.display(), err))
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
        if settings.enforce_reduce_only { "1" } else { "0" },
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

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn live_connector_supported(connector: ConnectorArg) -> bool {
    match connector {
        ConnectorArg::Hyperliquid | ConnectorArg::HyperliquidFixture => cfg!(feature = "live_hyperliquid"),
        ConnectorArg::Lighter => cfg!(feature = "live_lighter"),
        ConnectorArg::Mock => true,
        ConnectorArg::Extended | ConnectorArg::Aster | ConnectorArg::Paradex => false,
    }
}

fn live_connector_allowed_for_live_mode(connector: ConnectorArg) -> bool {
    matches!(connector, ConnectorArg::Hyperliquid | ConnectorArg::Lighter)
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

fn enforce_live_execution_guardrails(
    args: &Args,
    trade_mode: TradeMode,
    connector: ConnectorArg,
    canary_profile: Option<&PathBuf>,
    canary_settings: Option<&CanarySettings>,
) {
    if trade_mode != TradeMode::Live {
        return;
    }

    let exec_enable_env = env_is_true("PARAPHINA_LIVE_EXEC_ENABLE");
    let exec_confirm_env = env_is_yes("PARAPHINA_LIVE_EXECUTION_CONFIRM");
    let live_flag = args.enable_live_execution;

    if !live_connector_allowed_for_live_mode(connector) {
        eprintln!(
            "paraphina_live | error=live_mode_connector_invalid connector={} (use --trade-mode shadow for safe runs)",
            connector.as_str()
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
    let reconcile_enabled = std::env::var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .map(|v| v > 0)
        .unwrap_or(false);
    if !reconcile_enabled {
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
    connector: ConnectorArg,
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

    let connector_ok = live_connector_supported(connector);
    let connector_detail = format!("selected={} supported={}", connector.as_str(), connector_ok);
    checks.push(PreflightCheck {
        label: "connector",
        ok: connector_ok,
        details: connector_detail,
    });

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
        let reconcile_ok = std::env::var("PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .map(|v| v > 0)
            .unwrap_or(false);
        checks.push(PreflightCheck {
            label: "reconciliation",
            ok: reconcile_ok,
            details: if reconcile_ok { "enabled" } else { "missing" }.to_string(),
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
            path.parent().map(|p| std::fs::create_dir_all(p).is_ok()).unwrap_or(true)
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
    match connector {
        ConnectorArg::Hyperliquid | ConnectorArg::HyperliquidFixture => {
            let key_present = env_present("HL_PRIVATE_KEY");
            let vault_present = env_present("HL_VAULT_ADDRESS");
            if trade_mode == TradeMode::Live {
                creds_ok = key_present && vault_present;
            }
            creds_detail = format!(
                "hl_private_key={} hl_vault_address={}",
                key_present, vault_present
            );
            if connector == ConnectorArg::HyperliquidFixture {
                let fixture_dir = std::env::var("HL_FIXTURE_DIR")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|_| std::path::PathBuf::from("./tests/fixtures/hyperliquid"));
                let fixture_ok = fixture_dir.is_dir();
                creds_ok = creds_ok && fixture_ok;
                creds_detail.push_str(&format!(" fixture_dir_ok={}", fixture_ok));
            }
        }
        ConnectorArg::Lighter => {
            creds_ok = true;
            creds_detail = "no_key_required=true".to_string();
        }
        ConnectorArg::Mock | ConnectorArg::Extended | ConnectorArg::Aster | ConnectorArg::Paradex => {
            creds_ok = trade_mode != TradeMode::Live;
            creds_detail = "no_live_keys".to_string();
        }
    }
    checks.push(PreflightCheck {
        label: "credentials",
        ok: creds_ok,
        details: creds_detail,
    });

    let live_guard_ok = if trade_mode == TradeMode::Live {
        live_connector_allowed_for_live_mode(connector)
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
    let connector = resolve_connector(args.connector);
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
    if args.preflight {
        let ok = run_preflight(
            &args,
            trade_mode.trade_mode,
            connector,
            out_dir.clone(),
            canary_error.as_deref(),
            canary_settings.as_ref(),
        );
        std::process::exit(if ok { 0 } else { 1 });
    }
    enforce_live_execution_guardrails(
        &args,
        trade_mode.trade_mode,
        connector,
        canary_profile.as_ref(),
        canary_settings.as_ref(),
    );
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

    let (market_ingest_tx, mut market_ingest_rx) =
        mpsc::channel::<paraphina::live::types::MarketDataEvent>(1024);
    let (market_tx, market_rx) = mpsc::channel::<paraphina::live::types::MarketDataEvent>(1024);
    let (paper_market_tx, paper_market_rx) = mpsc::channel::<PaperMarketUpdate>(1024);
    let paper_market_tx = if paper_mode { Some(paper_market_tx) } else { None };
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

    if matches!(connector, ConnectorArg::Mock) {
        let market_tx_clone = market_ingest_tx.clone();
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

    let exec_enable_env = env_is_true("PARAPHINA_LIVE_EXEC_ENABLE");
    let allow_live_gateway = exec_enable_env
        && trade_mode.trade_mode != TradeMode::Shadow
        && (!paper_mode || paper_route_sandbox);
    let mut exec_client: Option<std::sync::Arc<dyn LiveRestClient>> = None;
    let mut exec_supported = false;

    match connector {
        ConnectorArg::Hyperliquid => {
            #[cfg(feature = "live_hyperliquid")]
            {
                let hl_cfg = paraphina::live::connectors::hyperliquid::HyperliquidConfig::from_env();
                let mut hl = paraphina::live::connectors::hyperliquid::HyperliquidConnector::new(
                    hl_cfg.clone(),
                    market_ingest_tx.clone(),
                    exec_tx.clone(),
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
                if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
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
                            feed.run_ticks(market_ingest_tx.clone(), 1_000, 250, 200).await;
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
                    market_ingest_tx.clone(),
                    exec_tx.clone(),
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
                if allow_live_gateway && trade_mode.trade_mode != TradeMode::Shadow {
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

    let exec_enabled = allow_live_gateway && exec_supported && exec_client.is_some();
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
    let exec_client = exec_client.clone();
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
                    eprintln!("paraphina_live | exec_gateway_error={} fallback=shadow", err.message);
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
    use super::ConnectorArg;
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
            ConnectorSupport::MarketAccount
        );
    }

    #[cfg(feature = "live_aster")]
    #[test]
    fn roadmap_b_feature_live_aster_enabled() {
        assert!(cfg!(feature = "live_aster"));
        assert_eq!(
            connector_support(ConnectorArg::Aster),
            ConnectorSupport::MarketAccount
        );
    }

    #[cfg(feature = "live_paradex")]
    #[test]
    fn roadmap_b_feature_live_paradex_enabled() {
        assert!(cfg!(feature = "live_paradex"));
        assert_eq!(
            connector_support(ConnectorArg::Paradex),
            ConnectorSupport::MarketAccount
        );
    }
}
