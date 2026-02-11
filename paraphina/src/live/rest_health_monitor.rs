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

/// Run the monitor loop.  Never returns (designed to be spawned supervised).
pub async fn run_rest_health_monitor(
    ages: SharedVenueAges,
    venues: Vec<VenueRestEntry>,
    market_tx: mpsc::Sender<MarketDataEvent>,
    cfg: RestMonitorConfig,
) {
    let mut interval = tokio::time::interval(cfg.poll_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let monitor_start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    // Track whether each venue was logged as active/inactive.
    let mut active: Vec<bool> = vec![false; venues.len()];

    loop {
        interval.tick().await;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        for (i, venue) in venues.iter().enumerate() {
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
            match tokio::time::timeout(fetch_timeout, (venue.fetcher)()).await {
                Ok(Ok(event)) => {
                    if market_tx.send(event).await.is_err() {
                        eprintln!(
                            "WARN: REST health monitor: market_tx closed for {}",
                            venue.name
                        );
                    }
                }
                Ok(Err(err)) => {
                    eprintln!(
                        "WARN: REST health monitor: {} REST fetch error: {err}",
                        venue.name
                    );
                }
                Err(_) => {
                    eprintln!(
                        "WARN: REST health monitor: {} REST fetch timed out ({}s)",
                        venue.name,
                        fetch_timeout.as_secs()
                    );
                }
            }
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
