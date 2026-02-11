//! Layer B — Centralised REST health monitor.
//!
//! Runs as an independent supervised task that periodically checks
//! [`SharedVenueAges`].  When a venue is stale beyond a configurable
//! threshold and has a REST API for L2 book data, the monitor fetches
//! the book via REST and sends the resulting `MarketDataEvent::L2Snapshot`
//! directly to the runner's `market_ingest_tx`, completely bypassing
//! connector-internal code.
//!
//! This layer survives connector bugs because it shares no state with
//! connectors — it reads ages from the runner and fetches data
//! independently.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;

use super::shared_venue_ages::SharedVenueAges;
use super::types::MarketDataEvent;

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

/// A type-erased async function that fetches a single L2 snapshot.
pub type RestFetcher = Box<dyn Fn() -> BoxFut + Send + Sync>;
type BoxFut =
    std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<MarketDataEvent>> + Send>>;

pub struct VenueRestEntry {
    /// Human-readable venue name for logging.
    pub name: String,
    /// Which venue index this fetcher covers.
    pub venue_index: usize,
    /// The async fetcher closure.
    pub fetcher: RestFetcher,
}

// ─── monitor loop ──────────────────────────────────────────────────────────

pub struct RestMonitorConfig {
    /// Age threshold (ms) beyond which REST polling kicks in.
    pub rest_threshold_ms: i64,
    /// How often the monitor checks ages.
    pub poll_interval: Duration,
}

impl Default for RestMonitorConfig {
    fn default() -> Self {
        Self {
            rest_threshold_ms: std::env::var("PARAPHINA_REST_MONITOR_THRESHOLD_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20_000),
            poll_interval: Duration::from_secs(5),
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
    last_log_ms: i64,
}

fn rest_monitor_ws_audit_enabled() -> bool {
    std::env::var("PARAPHINA_WS_AUDIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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
    let should_log = stats.rest_check_count <= 3 || now_ms.saturating_sub(stats.last_log_ms) >= 30_000;
    if !should_log {
        return;
    }
    stats.last_log_ms = now_ms;
    eprintln!(
        "WS_AUDIT subsystem=rest_monitor venue={} rest_check_count={} rest_attempt_count={} rest_success_count={} rest_fail_count={} rest_inject_count={} age_ms={} threshold_ms={}",
        venue.name,
        stats.rest_check_count,
        stats.rest_attempt_count,
        stats.rest_success_count,
        stats.rest_fail_count,
        stats.rest_inject_count,
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
    let monitor_start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    // Track whether each venue was logged as active/inactive.
    let mut active: Vec<bool> = vec![false; venues.len()];
    let mut audit: Vec<VenueRestAuditStats> = vec![VenueRestAuditStats::default(); venues.len()];

    loop {
        interval.tick().await;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        for (i, venue) in venues.iter().enumerate() {
            let stats = &mut audit[i];
            stats.rest_check_count += 1;
            let raw_age = ages.age_ms(venue.venue_index);
            // i64::MAX means "unknown/uninitialized"; avoid instant startup fallback.
            // Treat unknown as elapsed time since monitor start so fallback can still
            // activate if no real updates arrive within threshold.
            let age = if raw_age == i64::MAX {
                (now_ms - monitor_start_ms).max(0)
            } else {
                raw_age
            };
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
                    "WARN: REST health monitor: {} stale (age_ms={}, threshold={}), activating REST fallback",
                    venue.name, age, cfg.rest_threshold_ms
                );
                active[i] = true;
            }

            // Fetch with a per-request timeout to avoid blocking the monitor.
            let fetch_timeout = Duration::from_secs(5);
            stats.rest_attempt_count += 1;
            match tokio::time::timeout(fetch_timeout, (venue.fetcher)()).await {
                Ok(Ok(event)) => {
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
) -> anyhow::Result<MarketDataEvent> {
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
    Ok(MarketDataEvent::L2Snapshot(L2Snapshot {
        venue_index,
        venue_id: market.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    }))
}

/// Fetch Lighter L2 book via REST.
/// URLs attempted (in order): `{rest_url}/api/v1/orderBooks`, `{rest_url}/api/v1/orderbooks`
#[cfg(feature = "live_lighter")]
pub async fn fetch_lighter_l2_snapshot(
    client: &Client,
    rest_url: &str,
    market: &str,
    venue_index: usize,
) -> anyhow::Result<MarketDataEvent> {
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
            Ok(event) => return Ok(event),
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
) -> anyhow::Result<MarketDataEvent> {
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
    Ok(MarketDataEvent::L2Snapshot(L2Snapshot {
        venue_index,
        venue_id: market.to_string(),
        seq,
        timestamp_ms,
        bids,
        asks,
    }))
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
) -> anyhow::Result<MarketDataEvent> {
    let url = format!("{rest_url}/orderbook/{market}?depth={depth}");
    let resp = client.get(&url).send().await?.error_for_status()?;
    let value: Value = resp.json().await?;
    let results = value.get("results").unwrap_or(&value);
    let bids = parse_string_pair_levels(results.get("bids"), "bids")?;
    let asks = parse_string_pair_levels(results.get("asks"), "asks")?;
    let seq = results.get("seq_no").and_then(|v| v.as_u64()).unwrap_or(0);
    let timestamp_ms = results
        .get("last_updated_at")
        .and_then(|v| v.as_i64())
        .unwrap_or_else(wall_ms);
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
