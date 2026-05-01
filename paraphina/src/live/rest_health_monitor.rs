//! Layer B — Centralised REST health monitor.
//!
//! Runs as an independent supervised task that periodically checks
//! [`SharedVenueAges`].  When a venue is stale beyond a configurable
//! threshold, the monitor can either fetch and inject a REST market event
//! or run a lightweight probe-only check, depending on venue semantics.
//!
//! This layer survives connector bugs because it shares no state with
//! connectors — it reads ages from the runner and fetches data
//! independently.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;

use super::shared_venue_ages::SharedVenueAges;
use super::types::MarketDataEvent;
#[cfg(feature = "live_aster")]
use crate::live::connectors::aster::{
    aster_note_probe_rate_limit, aster_probe_budget_decision, AsterProbeBudgetDecision,
    AsterPublicRestBudgetHandle,
};

// reqwest-dependent REST fetch functions are only available when at least one
// connector feature pulls in the reqwest crate.  Some items may appear unused
// depending on which specific connector features are enabled.
#[cfg(any(
    feature = "live_lighter",
    feature = "live_aster",
    feature = "live_extended",
    feature = "live_paradex",
))]
#[allow(unused_imports)]
use {
    super::orderbook_l2::BookLevel, super::types::L2Snapshot, crate::types::TimestampMs,
    reqwest::Client, serde_json::Value,
};

// ─── per-venue REST fetcher trait ──────────────────────────────────────────

/// A type-erased async function that either returns an injectable market event
/// or reports successful probe completion with `None`.
pub type RestFetcher = Box<dyn Fn() -> BoxFut + Send + Sync>;
type BoxFut =
    std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<RestFetchOutcome>> + Send>>;

#[derive(Debug)]
pub enum RestFetchOutcome {
    Inject(MarketDataEvent),
    ProbeOk,
    Suppressed { reason: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestMonitorMode {
    InjectSnapshot,
    ProbeOnly,
}

impl RestMonitorMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::InjectSnapshot => "inject_snapshot",
            Self::ProbeOnly => "probe_only",
        }
    }
}

pub struct VenueRestEntry {
    /// Human-readable venue name for logging.
    pub name: String,
    /// Which venue index this fetcher covers.
    pub venue_index: usize,
    /// Whether a successful fetch should be injected or treated as a probe.
    pub mode: RestMonitorMode,
    /// The async fetcher closure.
    pub fetcher: RestFetcher,
}

// ─── monitor loop ──────────────────────────────────────────────────────────

pub struct RestMonitorConfig {
    /// Age threshold (ms) beyond which REST polling kicks in.
    pub rest_threshold_ms: i64,
    /// How often the monitor checks ages.
    pub poll_interval: Duration,
    /// One-shot startup seed delay (ms) for inject-snapshot venues.
    pub startup_seed_delay_ms: i64,
}

impl Default for RestMonitorConfig {
    fn default() -> Self {
        Self {
            rest_threshold_ms: std::env::var("PARAPHINA_REST_MONITOR_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20_000),
            poll_interval: Duration::from_secs(5),
            startup_seed_delay_ms: std::env::var("PARAPHINA_REST_MONITOR_STARTUP_SEED_DELAY_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v: &i64| *v > 0)
                .unwrap_or(1_000),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct VenueRestAuditStats {
    rest_check_count: u64,
    rest_attempt_count: u64,
    rest_success_count: u64,
    rest_fail_count: u64,
    rest_inject_count: u64,
    rest_suppressed_count: u64,
    last_log_ms: i64,
}

fn rest_monitor_ws_audit_enabled() -> bool {
    std::env::var("PARAPHINA_WS_AUDIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(feature = "live_paradex")]
fn paradex_rest_backstop_wall_ts_enabled() -> bool {
    std::env::var("PARAPHINA_PARADEX_REST_BACKSTOP_WALL_TS_ENABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true)
}

fn maybe_log_rest_audit(
    enabled: bool,
    now_ms: i64,
    venue: &VenueRestEntry,
    age_ms: i64,
    threshold_ms: i64,
    stats: &mut VenueRestAuditStats,
) {
    if !enabled {
        return;
    }
    let should_log =
        stats.rest_check_count <= 3 || now_ms.saturating_sub(stats.last_log_ms) >= 30_000;
    if !should_log {
        return;
    }
    stats.last_log_ms = now_ms;
    eprintln!(
        "WS_AUDIT subsystem=rest_monitor venue={} mode={} rest_check_count={} rest_attempt_count={} rest_success_count={} rest_fail_count={} rest_inject_count={} rest_suppressed_count={} age_ms={} threshold_ms={}",
        venue.name,
        venue.mode.as_str(),
        stats.rest_check_count,
        stats.rest_attempt_count,
        stats.rest_success_count,
        stats.rest_fail_count,
        stats.rest_inject_count,
        stats.rest_suppressed_count,
        age_ms,
        threshold_ms
    );
}

/// Run the monitor loop.  Never returns (designed to be spawned supervised).
pub async fn run_rest_health_monitor(
    ages: SharedVenueAges,
    venues: Vec<VenueRestEntry>,
    market_tx: mpsc::Sender<MarketDataEvent>,
    cfg: RestMonitorConfig,
) {
    let ws_audit_enabled = rest_monitor_ws_audit_enabled();
    let mut interval = tokio::time::interval(cfg.poll_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let startup_seed_enabled = cfg.startup_seed_delay_ms > 0;
    let mut startup_seed_interval = tokio::time::interval(Duration::from_millis(200));
    startup_seed_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let monitor_start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    // Track whether each venue was logged as active/inactive.
    let mut active: Vec<bool> = vec![false; venues.len()];
    let mut audit: Vec<VenueRestAuditStats> = vec![VenueRestAuditStats::default(); venues.len()];
    let mut startup_seed_done: Vec<bool> = venues
        .iter()
        .map(|venue| !matches!(venue.mode, RestMonitorMode::InjectSnapshot))
        .collect();

    loop {
        let startup_seed_pending =
            startup_seed_enabled && startup_seed_done.iter().any(|done| !*done);
        let startup_seed_tick = if startup_seed_pending {
            tokio::select! {
                _ = interval.tick() => false,
                _ = startup_seed_interval.tick() => true,
            }
        } else {
            interval.tick().await;
            false
        };
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        for (i, venue) in venues.iter().enumerate() {
            let stats = &mut audit[i];
            let raw_age = ages.age_ms(venue.venue_index);
            // i64::MAX means "unknown/uninitialized"; avoid instant startup fallback.
            // Treat unknown as elapsed time since monitor start so fallback can still
            // activate if no real updates arrive within threshold.
            let age = if raw_age == i64::MAX {
                (now_ms - monitor_start_ms).max(0)
            } else {
                raw_age
            };

            if startup_seed_tick {
                if startup_seed_done[i] {
                    continue;
                }
                if raw_age != i64::MAX && age < cfg.startup_seed_delay_ms {
                    startup_seed_done[i] = true;
                    if ws_audit_enabled {
                        eprintln!(
                            "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_suppressed=1 suppression_reason=already_fresh delay_ms={} age_ms={}",
                            venue.name,
                            venue.mode.as_str(),
                            cfg.startup_seed_delay_ms,
                            age,
                        );
                    }
                    continue;
                }
                if age < cfg.startup_seed_delay_ms {
                    continue;
                }
                startup_seed_done[i] = true;
                if ws_audit_enabled {
                    eprintln!(
                        "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_attempted=1 delay_ms={} age_ms={}",
                        venue.name,
                        venue.mode.as_str(),
                        cfg.startup_seed_delay_ms,
                        age,
                    );
                }
                let fetch_timeout = Duration::from_secs(5);
                stats.rest_attempt_count += 1;
                match tokio::time::timeout(fetch_timeout, (venue.fetcher)()).await {
                    Ok(Ok(outcome)) => match outcome {
                        RestFetchOutcome::Inject(event) => {
                            stats.rest_success_count += 1;
                            if market_tx.send(event).await.is_err() {
                                stats.rest_fail_count += 1;
                                eprintln!(
                                    "WARN: REST health monitor: market_tx closed for {} during startup seed",
                                    venue.name
                                );
                                if ws_audit_enabled {
                                    eprintln!(
                                        "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_failed=1 reason=market_tx_closed delay_ms={} age_ms={}",
                                        venue.name,
                                        venue.mode.as_str(),
                                        cfg.startup_seed_delay_ms,
                                        age,
                                    );
                                }
                            } else {
                                stats.rest_inject_count += 1;
                                if ws_audit_enabled {
                                    eprintln!(
                                        "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_injected=1 delay_ms={} age_ms={}",
                                        venue.name,
                                        venue.mode.as_str(),
                                        cfg.startup_seed_delay_ms,
                                        age,
                                    );
                                }
                            }
                        }
                        RestFetchOutcome::ProbeOk => {
                            stats.rest_success_count += 1;
                            if ws_audit_enabled {
                                eprintln!(
                                    "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_suppressed=1 suppression_reason=probe_only_result delay_ms={} age_ms={}",
                                    venue.name,
                                    venue.mode.as_str(),
                                    cfg.startup_seed_delay_ms,
                                    age,
                                );
                            }
                        }
                        RestFetchOutcome::Suppressed { reason } => {
                            stats.rest_suppressed_count += 1;
                            if ws_audit_enabled {
                                eprintln!(
                                    "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_suppressed=1 suppression_reason={} delay_ms={} age_ms={}",
                                    venue.name,
                                    venue.mode.as_str(),
                                    reason,
                                    cfg.startup_seed_delay_ms,
                                    age,
                                );
                            }
                        }
                    },
                    Ok(Err(err)) => {
                        stats.rest_fail_count += 1;
                        eprintln!(
                            "WARN: REST health monitor: {} startup seed fetch error: {err}",
                            venue.name
                        );
                        if ws_audit_enabled {
                            eprintln!(
                                "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_failed=1 reason=fetch_error delay_ms={} age_ms={}",
                                venue.name,
                                venue.mode.as_str(),
                                cfg.startup_seed_delay_ms,
                                age,
                            );
                        }
                    }
                    Err(_) => {
                        stats.rest_fail_count += 1;
                        eprintln!(
                            "WARN: REST health monitor: {} startup seed timed out ({}s)",
                            venue.name,
                            fetch_timeout.as_secs()
                        );
                        if ws_audit_enabled {
                            eprintln!(
                                "WS_AUDIT subsystem=rest_monitor venue={} mode={} startup_seed_failed=1 reason=timeout delay_ms={} age_ms={}",
                                venue.name,
                                venue.mode.as_str(),
                                cfg.startup_seed_delay_ms,
                                age,
                            );
                        }
                    }
                }
                maybe_log_rest_audit(
                    ws_audit_enabled,
                    now_ms,
                    venue,
                    age,
                    cfg.rest_threshold_ms,
                    stats,
                );
                continue;
            }

            stats.rest_check_count += 1;
            if age < cfg.rest_threshold_ms {
                if active[i] {
                    eprintln!(
                        "INFO: REST health monitor: {} recovered (age_ms={})",
                        venue.name, age
                    );
                    active[i] = false;
                }
                maybe_log_rest_audit(
                    ws_audit_enabled,
                    now_ms,
                    venue,
                    age,
                    cfg.rest_threshold_ms,
                    stats,
                );
                continue;
            }

            if !active[i] {
                eprintln!(
                    "WARN: REST health monitor: {} stale (age_ms={}, threshold={}), activating REST {}",
                    venue.name,
                    age,
                    cfg.rest_threshold_ms,
                    venue.mode.as_str()
                );
                active[i] = true;
            }

            // Fetch with a per-request timeout to avoid blocking the monitor.
            let fetch_timeout = Duration::from_secs(5);
            stats.rest_attempt_count += 1;
            match tokio::time::timeout(fetch_timeout, (venue.fetcher)()).await {
                Ok(Ok(outcome)) => match outcome {
                    RestFetchOutcome::Inject(event) => {
                        stats.rest_success_count += 1;
                        if market_tx.send(event).await.is_err() {
                            stats.rest_fail_count += 1;
                            eprintln!(
                                "WARN: REST health monitor: market_tx closed for {}",
                                venue.name
                            );
                        } else {
                            stats.rest_inject_count += 1;
                        }
                    }
                    RestFetchOutcome::ProbeOk => {
                        stats.rest_success_count += 1;
                    }
                    RestFetchOutcome::Suppressed { reason } => {
                        stats.rest_suppressed_count += 1;
                        if ws_audit_enabled {
                            eprintln!(
                                "WS_AUDIT subsystem=rest_monitor venue={} mode={} rest_probe_suppressed=1 suppression_reason={}",
                                venue.name,
                                venue.mode.as_str(),
                                reason,
                            );
                        }
                    }
                },
                Ok(Err(err)) => {
                    stats.rest_fail_count += 1;
                    eprintln!(
                        "WARN: REST health monitor: {} REST fetch error: {err}",
                        venue.name
                    );
                }
                Err(_) => {
                    stats.rest_fail_count += 1;
                    eprintln!(
                        "WARN: REST health monitor: {} REST fetch timed out ({}s)",
                        venue.name,
                        fetch_timeout.as_secs()
                    );
                }
            }
            maybe_log_rest_audit(
                ws_audit_enabled,
                now_ms,
                venue,
                age,
                cfg.rest_threshold_ms,
                stats,
            );
        }
    }
}

// ─── standalone REST fetch functions ───────────────────────────────────────
// These are self-contained helpers that the paraphina_live.rs startup code
// wraps into `RestFetcher` closures.
//
// Everything below requires `reqwest` which is only available when at least
// one live_* connector feature is enabled.

/// Fetch Extended L2 book via REST.
/// URL: `{rest_url}/fapi/v1/depth?symbol={market}&limit={depth_limit}`
#[cfg(feature = "live_extended")]
pub async fn fetch_extended_l2_snapshot(
    client: &Client,
    rest_url: &str,
    market: &str,
    depth_limit: usize,
    venue_index: usize,
) -> anyhow::Result<RestFetchOutcome> {
    let url = format!("{rest_url}/fapi/v1/depth?symbol={market}&limit={depth_limit}");
    let resp = client.get(&url).send().await?.error_for_status()?;
    let value: Value = resp.json().await?;
    let bids = parse_binance_levels(value.get("bids"), "bids")?;
    let asks = parse_binance_levels(value.get("asks"), "asks")?;
    let seq = value
        .get("lastUpdateId")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let timestamp_ms = wall_ms();
    Ok(RestFetchOutcome::Inject(MarketDataEvent::L2Snapshot(
        L2Snapshot {
            venue_index,
            venue_id: market.to_string(),
            seq,
            timestamp_ms,
            bids,
            asks,
        },
    )))
}

/// Fetch Lighter L2 book via REST.
/// URLs attempted (in order): `{rest_url}/api/v1/orderBooks`, `{rest_url}/api/v1/orderbooks`
#[cfg(feature = "live_lighter")]
pub async fn fetch_lighter_l2_snapshot(
    client: &Client,
    rest_url: &str,
    market: &str,
    venue_index: usize,
) -> anyhow::Result<RestFetchOutcome> {
    let base = rest_url.trim_end_matches('/');
    let endpoints = ["/api/v1/orderBooks", "/api/v1/orderbooks"];
    let mut last_error: Option<String> = None;
    for endpoint in endpoints {
        let url = format!("{base}{endpoint}");
        let response = match client.get(&url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                last_error = Some(format!("request error url={url} err={err}"));
                continue;
            }
        };
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            let snippet: String = body.chars().take(160).collect();
            last_error = Some(format!(
                "non-success status={} url={} snippet={}",
                status, url, snippet
            ));
            continue;
        }
        let value: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(err) => {
                last_error = Some(format!("json parse error url={url} err={err}"));
                continue;
            }
        };
        match parse_lighter_snapshot_response(&value, market, venue_index) {
            Ok(event) => return Ok(RestFetchOutcome::Inject(event)),
            Err(err) => {
                last_error = Some(format!("parse snapshot error url={url} err={err}"));
                continue;
            }
        }
    }
    anyhow::bail!(
        "lighter REST snapshot fetch failed market={} reason={}",
        market,
        last_error.unwrap_or_else(|| "unknown".to_string())
    )
}

/// Fetch Aster L2 book via REST.
/// URL: `{rest_url}/fapi/v1/depth?symbol={market}&limit={depth_limit}`
#[cfg(feature = "live_aster")]
pub async fn fetch_aster_l2_snapshot(
    client: &Client,
    rest_url: &str,
    market: &str,
    depth_limit: usize,
    venue_index: usize,
) -> anyhow::Result<RestFetchOutcome> {
    let url = format!("{rest_url}/fapi/v1/depth?symbol={market}&limit={depth_limit}");
    let resp = client.get(&url).send().await?.error_for_status()?;
    let value: Value = resp.json().await?;
    let bids = parse_binance_levels(value.get("bids"), "bids")?;
    let asks = parse_binance_levels(value.get("asks"), "asks")?;
    let seq = value
        .get("lastUpdateId")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let timestamp_ms = wall_ms();
    Ok(RestFetchOutcome::Inject(MarketDataEvent::L2Snapshot(
        L2Snapshot {
            venue_index,
            venue_id: market.to_string(),
            seq,
            timestamp_ms,
            bids,
            asks,
        },
    )))
}

/// Probe Aster market-data REST without competing for full depth snapshots.
/// URL: `{rest_url}/fapi/v1/ticker/bookTicker?symbol={market}`
#[cfg(feature = "live_aster")]
pub async fn probe_aster_book_ticker(
    client: &Client,
    rest_url: &str,
    market: &str,
) -> anyhow::Result<RestFetchOutcome> {
    let url = format!("{rest_url}/fapi/v1/ticker/bookTicker?symbol={market}");
    let resp = client.get(&url).send().await?.error_for_status()?;
    let value: Value = resp.json().await?;
    let symbol = value
        .get("symbol")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing symbol"))?;
    if symbol != market {
        anyhow::bail!(
            "bookTicker symbol mismatch expected={} actual={}",
            market,
            symbol
        );
    }
    let bid = value
        .get("bidPrice")
        .and_then(|v| v.as_str())
        .and_then(|raw| raw.parse::<f64>().ok())
        .ok_or_else(|| anyhow::anyhow!("missing bidPrice"))?;
    let ask = value
        .get("askPrice")
        .and_then(|v| v.as_str())
        .and_then(|raw| raw.parse::<f64>().ok())
        .ok_or_else(|| anyhow::anyhow!("missing askPrice"))?;
    if !(bid.is_finite() && ask.is_finite() && ask >= bid) {
        anyhow::bail!("invalid bid/ask bid={} ask={}", bid, ask);
    }
    Ok(RestFetchOutcome::ProbeOk)
}

/// Probe Aster market-data REST, but yield to connector-owned recovery and shared cooldown.
#[cfg(feature = "live_aster")]
pub async fn probe_aster_book_ticker_budgeted(
    client: &Client,
    rest_url: &str,
    market: &str,
    budget: &AsterPublicRestBudgetHandle,
) -> anyhow::Result<RestFetchOutcome> {
    match aster_probe_budget_decision(budget) {
        AsterProbeBudgetDecision::Allow => {}
        AsterProbeBudgetDecision::Suppressed { reason, .. } => {
            return Ok(RestFetchOutcome::Suppressed { reason });
        }
    }
    let url = format!("{rest_url}/fapi/v1/ticker/bookTicker?symbol={market}");
    let resp = client.get(&url).send().await?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let weight_1m = headers.iter().find_map(|(name, value)| {
        if name.as_str().eq_ignore_ascii_case("x-mbx-used-weight-1m") {
            value.to_str().ok().and_then(|raw| raw.parse::<u64>().ok())
        } else {
            None
        }
    });
    if !status.is_success() {
        if matches!(status.as_u16(), 418 | 429) {
            let budget_state = aster_note_probe_rate_limit(budget, status.as_u16(), weight_1m);
            anyhow::bail!(
                "HTTP status {} for url ({}) consumer=probe priority=optional shared_cooldown_ms={} shared_weight_1m={}",
                status.as_u16(),
                url,
                budget_state.shared_cooldown_ms,
                budget_state
                    .shared_weight_1m
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "0".to_string())
            );
        }
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(160).collect();
        anyhow::bail!(
            "HTTP status {} for url ({}) consumer=probe priority=optional snippet={}",
            status.as_u16(),
            url,
            snippet
        );
    }
    let value: Value = resp.json().await?;
    let symbol = value
        .get("symbol")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing symbol"))?;
    if symbol != market {
        anyhow::bail!(
            "bookTicker symbol mismatch expected={} actual={}",
            market,
            symbol
        );
    }
    let bid = value
        .get("bidPrice")
        .and_then(|v| v.as_str())
        .and_then(|raw| raw.parse::<f64>().ok())
        .ok_or_else(|| anyhow::anyhow!("missing bidPrice"))?;
    let ask = value
        .get("askPrice")
        .and_then(|v| v.as_str())
        .and_then(|raw| raw.parse::<f64>().ok())
        .ok_or_else(|| anyhow::anyhow!("missing askPrice"))?;
    if !(bid.is_finite() && ask.is_finite() && ask >= bid) {
        anyhow::bail!("invalid bid/ask bid={} ask={}", bid, ask);
    }
    Ok(RestFetchOutcome::ProbeOk)
}

/// Fetch Paradex L2 book via REST.
/// URL: `{rest_url}/orderbook/{market}?depth={depth}`
#[cfg(feature = "live_paradex")]
pub async fn fetch_paradex_l2_snapshot(
    client: &Client,
    rest_url: &str,
    market: &str,
    depth: usize,
    venue_index: usize,
) -> anyhow::Result<RestFetchOutcome> {
    let url = format!("{rest_url}/orderbook/{market}?depth={depth}");
    let received_ms = wall_ms();
    let resp = client.get(&url).send().await?.error_for_status()?;
    let value: Value = resp.json().await?;
    Ok(RestFetchOutcome::Inject(
        build_paradex_l2_snapshot_from_value(&value, market, venue_index, received_ms)?,
    ))
}

#[cfg(feature = "live_paradex")]
fn build_paradex_l2_snapshot_from_value(
    value: &Value,
    market: &str,
    venue_index: usize,
    received_ms: TimestampMs,
) -> anyhow::Result<MarketDataEvent> {
    let results = value.get("results").unwrap_or(&value);
    let bids = parse_string_pair_levels(results.get("bids"), "bids")?;
    let asks = parse_string_pair_levels(results.get("asks"), "asks")?;
    let seq = results.get("seq_no").and_then(|v| v.as_u64()).unwrap_or(0);
    let exchange_timestamp_ms = results
        .get("last_updated_at")
        .and_then(|v| v.as_i64())
        .unwrap_or(received_ms);
    let timestamp_ms = if paradex_rest_backstop_wall_ts_enabled() {
        received_ms
    } else {
        exchange_timestamp_ms
    };
    Ok(MarketDataEvent::L2Snapshot(L2Snapshot {
        venue_index,
        venue_id: market.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    }))
}

// ─── helpers ───────────────────────────────────────────────────────────────

#[cfg(any(
    feature = "live_lighter",
    feature = "live_aster",
    feature = "live_extended",
    feature = "live_paradex",
))]
fn wall_ms() -> TimestampMs {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as TimestampMs
}

#[cfg(feature = "live_lighter")]
fn parse_lighter_snapshot_response(
    value: &Value,
    market: &str,
    venue_index: usize,
) -> anyhow::Result<MarketDataEvent> {
    let entries = value
        .as_array()
        .or_else(|| value.get("data").and_then(|v| v.as_array()))
        .or_else(|| value.get("order_books").and_then(|v| v.as_array()))
        .or_else(|| value.get("orderBooks").and_then(|v| v.as_array()))
        .ok_or_else(|| anyhow::anyhow!("missing orderBooks array"))?;
    let matched = entries
        .iter()
        .find(|entry| {
            entry
                .get("symbol")
                .or_else(|| entry.get("market"))
                .and_then(|v| v.as_str())
                .map(|symbol| lighter_symbol_matches(symbol, market))
                .unwrap_or(false)
        })
        .or_else(|| entries.first())
        .ok_or_else(|| anyhow::anyhow!("empty orderBooks array"))?;
    let symbol = matched
        .get("symbol")
        .or_else(|| matched.get("market"))
        .and_then(|v| v.as_str())
        .unwrap_or(market)
        .to_string();
    let book = matched
        .get("order_book")
        .or_else(|| matched.get("orderBook"))
        .unwrap_or(matched);
    let bids = parse_lighter_levels(book.get("bids"), "bids")?;
    let asks = parse_lighter_levels(book.get("asks"), "asks")?;
    let seq = book
        .get("seq")
        .or_else(|| book.get("sequence"))
        .or_else(|| book.get("lastUpdateId"))
        .or_else(|| matched.get("seq"))
        .or_else(|| matched.get("sequence"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let timestamp_ms = book
        .get("timestamp")
        .or_else(|| book.get("ts"))
        .or_else(|| book.get("updated_at"))
        .or_else(|| book.get("last_updated_at"))
        .or_else(|| matched.get("timestamp"))
        .or_else(|| matched.get("ts"))
        .and_then(|v| v.as_i64())
        .unwrap_or_else(wall_ms);
    Ok(MarketDataEvent::L2Snapshot(L2Snapshot {
        venue_index,
        venue_id: symbol,
        seq,
        timestamp_ms,
        bids,
        asks,
    }))
}

#[cfg(feature = "live_lighter")]
fn lighter_symbol_matches(symbol: &str, market: &str) -> bool {
    if symbol.eq_ignore_ascii_case(market) {
        return true;
    }
    fn normalize(s: &str) -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase()
    }
    normalize(symbol) == normalize(market)
}

#[cfg(feature = "live_lighter")]
fn parse_lighter_levels(value: Option<&Value>, label: &str) -> anyhow::Result<Vec<BookLevel>> {
    let arr = value
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing or invalid {label} array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        if let Some(obj) = entry.as_object() {
            let price = parse_lighter_str_or_number(
                obj.get("price").or_else(|| obj.get("px")),
                label,
                "price",
            )?;
            let size = parse_lighter_str_or_number(
                obj.get("size").or_else(|| obj.get("sz")),
                label,
                "size",
            )?;
            out.push(BookLevel { price, size });
            continue;
        }
        let pair = entry
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid {label} entry"))?;
        let price = parse_lighter_str_or_number(pair.first(), label, "price")?;
        let size = parse_lighter_str_or_number(pair.get(1), label, "size")?;
        out.push(BookLevel { price, size });
    }
    Ok(out)
}

#[cfg(feature = "live_lighter")]
fn parse_lighter_str_or_number(v: Option<&Value>, label: &str, field: &str) -> anyhow::Result<f64> {
    let v = v.ok_or_else(|| anyhow::anyhow!("{label} {field} missing"))?;
    if let Some(s) = v.as_str() {
        Ok(s.parse()?)
    } else if let Some(n) = v.as_f64() {
        Ok(n)
    } else if let Some(n) = v.as_i64() {
        Ok(n as f64)
    } else if let Some(n) = v.as_u64() {
        Ok(n as f64)
    } else {
        anyhow::bail!("{label} {field} is neither string nor number")
    }
}

/// Parse Binance-style levels: `[["price_str", "size_str"], ...]`
#[cfg(any(feature = "live_aster", feature = "live_extended"))]
fn parse_binance_levels(value: Option<&Value>, label: &str) -> anyhow::Result<Vec<BookLevel>> {
    let arr = value
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing or invalid {label} array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let pair = entry
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid {label} entry"))?;
        let price: f64 = pair
            .first()
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("{label} price not a string"))?
            .parse()?;
        let size: f64 = pair
            .get(1)
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("{label} size not a string"))?
            .parse()?;
        out.push(BookLevel { price, size });
    }
    Ok(out)
}

/// Parse Paradex-style levels: `[["price_str", "size_str"], ...]` where
/// values may be strings.
#[cfg(feature = "live_paradex")]
fn parse_string_pair_levels(value: Option<&Value>, label: &str) -> anyhow::Result<Vec<BookLevel>> {
    let arr = value
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing or invalid {label} array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let pair = entry
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid {label} entry"))?;
        let price: f64 = parse_str_or_number(pair.first(), label, "price")?;
        let size: f64 = parse_str_or_number(pair.get(1), label, "size")?;
        out.push(BookLevel { price, size });
    }
    Ok(out)
}

#[cfg(feature = "live_paradex")]
fn parse_str_or_number(v: Option<&Value>, label: &str, field: &str) -> anyhow::Result<f64> {
    let v = v.ok_or_else(|| anyhow::anyhow!("{label} {field} missing"))?;
    if let Some(s) = v.as_str() {
        Ok(s.parse()?)
    } else if let Some(n) = v.as_f64() {
        Ok(n)
    } else {
        anyhow::bail!("{label} {field} is neither string nor number")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live::shared_venue_ages::SharedVenueAges;
    use crate::live::types::{L2Snapshot, MarketDataEvent};
    use crate::orderbook_l2::BookLevel;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::sync::mpsc::error::TryRecvError;

    #[cfg(feature = "live_aster")]
    use httpmock::Method::GET;
    #[cfg(feature = "live_aster")]
    use httpmock::MockServer;

    #[cfg(feature = "live_aster")]
    #[tokio::test]
    async fn probe_aster_book_ticker_accepts_valid_payload() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/fapi/v1/ticker/bookTicker")
                    .query_param("symbol", "ETHUSDT");
                then.status(200).json_body(serde_json::json!({
                    "symbol": "ETHUSDT",
                    "bidPrice": "2163.10",
                    "bidQty": "1.25",
                    "askPrice": "2163.20",
                    "askQty": "0.75",
                    "time": 1774353533000u64
                }));
            })
            .await;
        let client = Client::builder().build().expect("client");

        let result = probe_aster_book_ticker(&client, &server.base_url(), "ETHUSDT")
            .await
            .expect("probe");

        mock.assert_async().await;
        assert!(matches!(result, RestFetchOutcome::ProbeOk));
    }

    #[cfg(feature = "live_paradex")]
    #[test]
    fn paradex_rest_backstop_uses_receive_time_for_snapshot_liveness() {
        let received_ms = wall_ms();
        let stale_exchange_ts = received_ms - 30_000;
        let value = serde_json::json!({
            "bids": [["3000.0", "0.25"]],
            "asks": [["3000.5", "0.20"]],
            "last_updated_at": stale_exchange_ts,
            "market": "ETH-USD-PERP",
            "seq_no": 77
        });

        let event = build_paradex_l2_snapshot_from_value(&value, "ETH-USD-PERP", 4, received_ms)
            .expect("paradex rest snapshot");

        match event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.venue_index, 4);
                assert_eq!(snapshot.venue_id, "ETH-USD-PERP");
                assert_eq!(snapshot.seq, 77);
                assert_eq!(snapshot.timestamp_ms, received_ms);
                assert_ne!(snapshot.timestamp_ms, stale_exchange_ts);
            }
            other => panic!("expected snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn probe_only_mode_does_not_inject_market_events() {
        let ages = SharedVenueAges::new(1);
        ages.set_age(0, 60_000);
        let (market_tx, mut market_rx) = mpsc::channel(4);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let venues = vec![VenueRestEntry {
            name: "aster".to_string(),
            venue_index: 0,
            mode: RestMonitorMode::ProbeOnly,
            fetcher: Box::new(move || {
                let calls = calls_clone.clone();
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(RestFetchOutcome::ProbeOk)
                })
            }),
        }];
        let monitor = tokio::spawn(run_rest_health_monitor(
            ages,
            venues,
            market_tx,
            RestMonitorConfig {
                rest_threshold_ms: 1,
                poll_interval: Duration::from_millis(10),
                startup_seed_delay_ms: 0,
            },
        ));

        tokio::time::sleep(Duration::from_millis(35)).await;
        assert!(
            calls.load(Ordering::Relaxed) > 0,
            "probe-only fetcher should still run for stale venues"
        );
        assert!(matches!(market_rx.try_recv(), Err(TryRecvError::Empty)));
        monitor.abort();
    }

    #[tokio::test]
    async fn startup_seed_injects_once_for_stale_inject_snapshot_venue() {
        let ages = SharedVenueAges::new(1);
        let (market_tx, mut market_rx) = mpsc::channel(4);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let venues = vec![VenueRestEntry {
            name: "paradex".to_string(),
            venue_index: 0,
            mode: RestMonitorMode::InjectSnapshot,
            fetcher: Box::new(move || {
                let calls = calls_clone.clone();
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(RestFetchOutcome::Inject(MarketDataEvent::L2Snapshot(
                        L2Snapshot {
                            venue_index: 0,
                            venue_id: "paradex".to_string(),
                            seq: 1,
                            timestamp_ms: 1_000,
                            bids: vec![BookLevel {
                                price: 2_000.0,
                                size: 1.0,
                            }],
                            asks: vec![BookLevel {
                                price: 2_000.5,
                                size: 1.0,
                            }],
                        },
                    )))
                })
            }),
        }];
        let monitor = tokio::spawn(run_rest_health_monitor(
            ages,
            venues,
            market_tx,
            RestMonitorConfig {
                rest_threshold_ms: 60_000,
                poll_interval: Duration::from_secs(60),
                startup_seed_delay_ms: 10,
            },
        ));

        let event = tokio::time::timeout(Duration::from_millis(250), market_rx.recv())
            .await
            .expect("startup seed should inject before timeout")
            .expect("market event");
        match event {
            MarketDataEvent::L2Snapshot(snapshot) => {
                assert_eq!(snapshot.venue_id, "paradex");
                assert_eq!(snapshot.seq, 1);
            }
            other => panic!("expected snapshot startup seed, got {other:?}"),
        }
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "startup seed should fire once"
        );
        monitor.abort();
    }

    #[tokio::test]
    async fn startup_seed_is_suppressed_when_venue_becomes_fresh_before_delay() {
        let ages = SharedVenueAges::new(1);
        let (market_tx, mut market_rx) = mpsc::channel(4);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let venues = vec![VenueRestEntry {
            name: "paradex".to_string(),
            venue_index: 0,
            mode: RestMonitorMode::InjectSnapshot,
            fetcher: Box::new(move || {
                let calls = calls_clone.clone();
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(RestFetchOutcome::Inject(MarketDataEvent::L2Snapshot(
                        L2Snapshot {
                            venue_index: 0,
                            venue_id: "paradex".to_string(),
                            seq: 1,
                            timestamp_ms: 1_000,
                            bids: vec![BookLevel {
                                price: 2_000.0,
                                size: 1.0,
                            }],
                            asks: vec![BookLevel {
                                price: 2_000.5,
                                size: 1.0,
                            }],
                        },
                    )))
                })
            }),
        }];
        let monitor = tokio::spawn(run_rest_health_monitor(
            ages.clone(),
            venues,
            market_tx,
            RestMonitorConfig {
                rest_threshold_ms: 60_000,
                poll_interval: Duration::from_secs(60),
                startup_seed_delay_ms: 50,
            },
        ));

        tokio::time::sleep(Duration::from_millis(10)).await;
        ages.set_age(0, 1);
        tokio::time::sleep(Duration::from_millis(120)).await;

        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "startup seed should be suppressed once venue becomes fresh"
        );
        assert!(matches!(market_rx.try_recv(), Err(TryRecvError::Empty)));
        monitor.abort();
    }
}
