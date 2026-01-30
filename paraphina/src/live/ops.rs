//! Operational controls for live trading (feature-gated).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use prometheus::{Encoder, IntCounter, IntCounterVec, IntGauge, Opts, Registry, TextEncoder};
use serde::Serialize;
use tiny_http::{Header, Response, Server};

use crate::config::Config;
use crate::live::instrument::InstrumentSpec;
use crate::live::state_cache::AccountReconcileDiff;
use crate::live::trade_mode::TradeMode;
use crate::sim_eval::BuildInfo;
use crate::telemetry::ReconcileDriftRecord;

pub trait SecretProvider: Send + Sync {
    fn get(&self, key: &str) -> Option<String>;
}

#[derive(Debug, Default)]
pub struct EnvSecretProvider;

impl SecretProvider for EnvSecretProvider {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

#[derive(Debug, Clone)]
pub struct HealthState {
    healthy: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
}

impl HealthState {
    pub fn new() -> Self {
        Self {
            healthy: Arc::new(AtomicBool::new(true)),
            ready: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
pub struct LiveMetrics {
    registry: Registry,
    tick_total: IntCounter,
    orders_sent: IntCounter,
    cancel_all_sent: IntCounter,
    errors_total: IntCounter,
    last_tick_ms: IntGauge,
    order_submit_ok: IntCounter,
    order_submit_fail: IntCounter,
    cancel_ok: IntCounter,
    cancel_fail: IntCounter,
    reject_by_reason: IntCounterVec,
    retry_count: IntCounter,
    rate_limit_sleep_total: IntCounter,
    reconcile_mismatch_count: IntCounter,
    market_rx_raw_drained_total: IntCounter,
    market_rx_out_emitted_total: IntCounter,
    market_rx_cap_hits_total: IntCounter,
}

impl LiveMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let tick_total = IntCounter::with_opts(Opts::new("paraphina_live_ticks", "Tick count"))
            .expect("tick counter");
        let orders_sent = IntCounter::with_opts(Opts::new("paraphina_live_orders", "Orders sent"))
            .expect("orders counter");
        let cancel_all_sent =
            IntCounter::with_opts(Opts::new("paraphina_live_cancel_all", "Cancel-all sent"))
                .expect("cancel-all counter");
        let errors_total =
            IntCounter::with_opts(Opts::new("paraphina_live_errors", "Errors total"))
                .expect("errors counter");
        let last_tick_ms = IntGauge::with_opts(Opts::new(
            "paraphina_live_last_tick_ms",
            "Last tick timestamp ms",
        ))
        .expect("tick gauge");
        let order_submit_ok = IntCounter::with_opts(Opts::new(
            "paraphina_live_order_submit_ok",
            "Order submit ok",
        ))
        .expect("order submit ok");
        let order_submit_fail = IntCounter::with_opts(Opts::new(
            "paraphina_live_order_submit_fail",
            "Order submit fail",
        ))
        .expect("order submit fail");
        let cancel_ok = IntCounter::with_opts(Opts::new("paraphina_live_cancel_ok", "Cancel ok"))
            .expect("cancel ok");
        let cancel_fail =
            IntCounter::with_opts(Opts::new("paraphina_live_cancel_fail", "Cancel fail"))
                .expect("cancel fail");
        let reject_by_reason = IntCounterVec::new(
            Opts::new("paraphina_live_reject_by_reason", "Rejects by reason"),
            &["reason"],
        )
        .expect("reject by reason");
        let retry_count =
            IntCounter::with_opts(Opts::new("paraphina_live_retry_count", "Retry count"))
                .expect("retry count");
        let rate_limit_sleep_total = IntCounter::with_opts(Opts::new(
            "paraphina_live_rate_limit_sleep_total_ms",
            "Rate limit sleep total (ms)",
        ))
        .expect("rate limit sleep");
        let reconcile_mismatch_count = IntCounter::with_opts(Opts::new(
            "paraphina_live_reconcile_mismatch_count",
            "Reconcile mismatch count",
        ))
        .expect("reconcile mismatch");
        let market_rx_raw_drained_total = IntCounter::with_opts(Opts::new(
            "paraphina_live_market_rx_raw_drained_total",
            "Market RX raw drained total",
        ))
        .expect("market rx raw drained");
        let market_rx_out_emitted_total = IntCounter::with_opts(Opts::new(
            "paraphina_live_market_rx_out_emitted_total",
            "Market RX out emitted total",
        ))
        .expect("market rx out emitted");
        let market_rx_cap_hits_total = IntCounter::with_opts(Opts::new(
            "paraphina_live_market_rx_cap_hits_total",
            "Market RX cap hits total",
        ))
        .expect("market rx cap hits");
        registry
            .register(Box::new(tick_total.clone()))
            .expect("reg tick");
        registry
            .register(Box::new(orders_sent.clone()))
            .expect("reg orders");
        registry
            .register(Box::new(cancel_all_sent.clone()))
            .expect("reg cancel");
        registry
            .register(Box::new(errors_total.clone()))
            .expect("reg errors");
        registry
            .register(Box::new(last_tick_ms.clone()))
            .expect("reg gauge");
        registry
            .register(Box::new(order_submit_ok.clone()))
            .expect("reg order submit ok");
        registry
            .register(Box::new(order_submit_fail.clone()))
            .expect("reg order submit fail");
        registry
            .register(Box::new(cancel_ok.clone()))
            .expect("reg cancel ok");
        registry
            .register(Box::new(cancel_fail.clone()))
            .expect("reg cancel fail");
        registry
            .register(Box::new(reject_by_reason.clone()))
            .expect("reg reject by reason");
        registry
            .register(Box::new(retry_count.clone()))
            .expect("reg retry count");
        registry
            .register(Box::new(rate_limit_sleep_total.clone()))
            .expect("reg rate limit sleep");
        registry
            .register(Box::new(reconcile_mismatch_count.clone()))
            .expect("reg reconcile mismatch");
        registry
            .register(Box::new(market_rx_raw_drained_total.clone()))
            .expect("reg market rx raw drained");
        registry
            .register(Box::new(market_rx_out_emitted_total.clone()))
            .expect("reg market rx out emitted");
        registry
            .register(Box::new(market_rx_cap_hits_total.clone()))
            .expect("reg market rx cap hits");
        Self {
            registry,
            tick_total,
            orders_sent,
            cancel_all_sent,
            errors_total,
            last_tick_ms,
            order_submit_ok,
            order_submit_fail,
            cancel_ok,
            cancel_fail,
            reject_by_reason,
            retry_count,
            rate_limit_sleep_total,
            reconcile_mismatch_count,
            market_rx_raw_drained_total,
            market_rx_out_emitted_total,
            market_rx_cap_hits_total,
        }
    }

    pub fn inc_tick(&self, now_ms: i64) {
        self.tick_total.inc();
        self.last_tick_ms.set(now_ms);
    }

    pub fn inc_orders(&self, count: usize) {
        self.orders_sent.inc_by(count as u64);
    }

    pub fn inc_cancel_all(&self) {
        self.cancel_all_sent.inc();
    }

    pub fn inc_error(&self) {
        self.errors_total.inc();
    }

    pub fn inc_order_submit_ok(&self) {
        self.order_submit_ok.inc();
    }

    pub fn inc_order_submit_fail(&self) {
        self.order_submit_fail.inc();
    }

    pub fn inc_cancel_ok(&self) {
        self.cancel_ok.inc();
    }

    pub fn inc_cancel_fail(&self) {
        self.cancel_fail.inc();
    }

    pub fn inc_reject_reason(&self, reason: &str) {
        self.reject_by_reason.with_label_values(&[reason]).inc();
    }

    pub fn inc_retry(&self) {
        self.retry_count.inc();
    }

    pub fn add_rate_limit_sleep_ms(&self, ms: u64) {
        self.rate_limit_sleep_total.inc_by(ms);
    }

    pub fn inc_reconcile_mismatch(&self) {
        self.reconcile_mismatch_count.inc();
    }

    pub fn add_market_rx_stats(&self, raw_drained: u64, out_emitted: u64, cap_hits: u64) {
        self.market_rx_raw_drained_total.inc_by(raw_drained);
        self.market_rx_out_emitted_total.inc_by(out_emitted);
        self.market_rx_cap_hits_total.inc_by(cap_hits);
    }

    pub fn gather(&self) -> String {
        let mf = self.registry.gather();
        let mut buf = Vec::new();
        let encoder = TextEncoder::new();
        let _ = encoder.encode(&mf, &mut buf);
        String::from_utf8(buf).unwrap_or_default()
    }
}

pub fn start_metrics_server(
    addr: &str,
    metrics: LiveMetrics,
    health: HealthState,
    audit_dir: PathBuf,
) {
    let addr = addr.to_string();
    std::thread::spawn(move || {
        let Ok(server) = Server::http(addr.as_str()) else {
            return;
        };
        for request in server.incoming_requests() {
            let url = request.url();
            let response = match url {
                "/config" => {
                    let config_path = audit_dir.join("config_resolved.json");
                    let build_path = audit_dir.join("build_info.json");
                    match (
                        fs::read_to_string(&config_path),
                        fs::read_to_string(&build_path),
                    ) {
                        (Ok(config), Ok(build)) => {
                            let payload = format!(
                                "{{\"config_resolved\":{},\"build_info\":{}}}",
                                config, build
                            );
                            Response::from_string(payload).with_header(
                                Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                                    .unwrap(),
                            )
                        }
                        _ => Response::from_string("config not found").with_status_code(404),
                    }
                }
                "/metrics" => Response::from_string(metrics.gather()).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"text/plain; version=0.0.4"[..])
                        .unwrap(),
                ),
                "/health" => {
                    if health.is_healthy() {
                        Response::from_string("ok")
                    } else {
                        Response::from_string("unhealthy").with_status_code(503)
                    }
                }
                "/ready" => {
                    if health.is_ready() {
                        Response::from_string("ready")
                    } else {
                        Response::from_string("not_ready").with_status_code(503)
                    }
                }
                _ => Response::from_string("not found").with_status_code(404),
            };
            let _ = request.respond(response);
        }
    });
}

#[derive(Debug, Serialize)]
struct AccountReconcileAuditRecord {
    now_ms: i64,
    diff: AccountReconcileDiff,
}

pub fn append_account_reconcile_audit(
    dir: &Path,
    now_ms: i64,
    diff: AccountReconcileDiff,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("account_reconcile.jsonl");
    let record = AccountReconcileAuditRecord { now_ms, diff };
    let line = serde_json::to_string(&record).unwrap_or_default();
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(format!("{line}\n").as_bytes())?;
    Ok(())
}

pub fn append_reconcile_drift_audit(
    dir: &Path,
    record: &ReconcileDriftRecord,
) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join("reconcile_drift.jsonl");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let line = serde_json::to_string(record).unwrap_or_else(|_| "{}".to_string());
    writeln!(file, "{line}")?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct LiveConfigResolved<'a> {
    config_version: &'a str,
    config_hash: String,
    config_debug: String,
}

pub fn config_hash(cfg: &Config) -> u64 {
    fnv1a64(&format!("{cfg:?}"))
}

pub fn format_startup_log(
    cfg: &Config,
    build_info: &BuildInfo,
    trade_mode: TradeMode,
    connectors_label: &str,
    metrics_addr: &str,
) -> String {
    let hash = config_hash(cfg);
    let venues = cfg
        .venues
        .iter()
        .map(|v| v.id.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let network = resolve_network_label();
    format!(
        "paraphina_live | trade_mode={} | network={} | venues={} | connectors={} | cfg={} | cfg_hash=0x{:016x} | build_id={} | dirty={} | intervals_ms=main:{} hedge:{} risk:{} | metrics_addr={}",
        trade_mode.as_str(),
        network,
        venues,
        connectors_label,
        cfg.version,
        hash,
        build_info.git_sha,
        build_info.dirty,
        cfg.main_loop_interval_ms,
        cfg.hedge_loop_interval_ms,
        cfg.risk_loop_interval_ms,
        metrics_addr
    )
}

pub fn resolve_network_label() -> String {
    if let Ok(val) = std::env::var("PARAPHINA_LIVE_NETWORK") {
        if !val.trim().is_empty() {
            return val;
        }
    }
    #[allow(unused_mut)]
    let mut labels: Vec<String> = Vec::new();
    #[cfg(feature = "live_hyperliquid")]
    {
        let cfg = crate::live::connectors::hyperliquid::HyperliquidConfig::from_env();
        let label = match cfg.network {
            crate::live::connectors::hyperliquid::HyperliquidNetwork::Mainnet => {
                "hyperliquid:mainnet"
            }
            crate::live::connectors::hyperliquid::HyperliquidNetwork::Testnet => {
                "hyperliquid:testnet"
            }
        };
        labels.push(label.to_string());
    }
    #[cfg(feature = "live_lighter")]
    {
        labels.push("lighter:default".to_string());
    }
    if labels.is_empty() {
        "unknown".to_string()
    } else {
        labels.join(",")
    }
}

pub fn resolve_connectors_label() -> String {
    #[allow(unused_mut)]
    let mut connectors: Vec<String> = Vec::new();
    #[cfg(feature = "live_hyperliquid")]
    {
        connectors.push("hyperliquid".to_string());
    }
    #[cfg(feature = "live_lighter")]
    {
        connectors.push("lighter".to_string());
    }
    if connectors.is_empty() {
        "none".to_string()
    } else {
        connectors.join(",")
    }
}

pub fn write_audit_files(dir: &Path, cfg: &Config, build_info: &BuildInfo) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let config_path = dir.join("config_resolved.json");
    let build_path = dir.join("build_info.json");
    let instrument_path = dir.join("instrument_specs.json");
    let resolved = LiveConfigResolved {
        config_version: cfg.version,
        config_hash: format!("0x{:016x}", config_hash(cfg)),
        config_debug: format!("{cfg:?}"),
    };
    let config_json = serde_json::to_string_pretty(&resolved)?;
    fs::write(config_path, config_json)?;
    crate::write_build_info(build_path, build_info)?;
    let specs = InstrumentSpec::from_config(cfg);
    fs::write(instrument_path, serde_json::to_string_pretty(&specs)?)?;
    Ok(())
}

fn fnv1a64(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

pub fn default_audit_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PARAPHINA_LIVE_AUDIT_DIR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(dir) = std::env::var("PARAPHINA_LIVE_OUT_DIR") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    PathBuf::from("./live_audit")
}
