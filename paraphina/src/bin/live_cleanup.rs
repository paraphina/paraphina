use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use paraphina::live::connectors::aster::{AsterConfig, AsterRestClient};
use paraphina::live::connectors::extended::{ExtendedConfig, ExtendedRestClient};
use paraphina::live::connectors::lighter::{LighterConfig, LighterConnector};
use paraphina::live::connectors::paradex::{ParadexConfig, ParadexRestClient};
use paraphina::live::gateway::{LiveRestCancelAllRequest, LiveRestClient, LiveRestPlaceRequest};
use paraphina::types::{OrderPurpose, Side, TimeInForce};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::sleep;

const VENUE_IDS: [&str; 5] = ["extended", "hyperliquid", "aster", "lighter", "paradex"];

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    env_file: PathBuf,
    #[arg(long, default_value = "/var/lib/paraphina/out/telemetry.jsonl")]
    telemetry_path: PathBuf,
    #[arg(long, default_value_t = 1024 * 1024)]
    tail_bytes: usize,
    #[arg(long, default_value_t = 100.0)]
    slippage_bps: f64,
    #[arg(long, default_value_t = 0.0)]
    lighter_pos: f64,
    #[arg(long, default_value_t = 0.0)]
    extended_pos: f64,
    #[arg(long, default_value_t = 0.0)]
    aster_pos: f64,
    #[arg(long, default_value_t = 0.0)]
    paradex_pos: f64,
    #[arg(long, default_value_t = 0)]
    settle_ms: u64,
    #[arg(long, default_value_t = false)]
    json_summary: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    load_env_file(&args.env_file)?;
    let mids = read_latest_mids(&args.telemetry_path, args.tail_bytes)?;

    let mut actions = Vec::new();
    actions.extend(cleanup_lighter(args.lighter_pos, args.slippage_bps, &mids).await?);
    actions.extend(cleanup_extended(args.extended_pos, args.slippage_bps, &mids).await?);
    actions.extend(cleanup_aster(args.aster_pos, args.slippage_bps, &mids).await?);
    actions.extend(cleanup_paradex(args.paradex_pos, args.slippage_bps, &mids).await?);

    if args.settle_ms > 0 {
        sleep(Duration::from_millis(args.settle_ms)).await;
    }

    let summary = build_cleanup_summary(actions, args.settle_ms);
    if args.json_summary {
        println!("{}", serde_json::to_string(&summary)?);
    } else {
        emit_cleanup_summary_logs(&summary);
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CleanupActionKind {
    CancelAll,
    ReduceOnlyIoc,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct CleanupAction {
    venue: String,
    kind: CleanupActionKind,
    side: Option<Side>,
    requested_size_base: Option<f64>,
    limit_price_usd: Option<f64>,
    reference_mid_usd: Option<f64>,
    slippage_bps: Option<f64>,
    estimated_cleanup_cost_usd: f64,
    order_id: Option<String>,
    status: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct CleanupSummary {
    result: String,
    settle_ms: u64,
    total_estimated_cleanup_cost_usd: f64,
    venues_touched: Vec<String>,
    actions: Vec<CleanupAction>,
}

fn load_env_file(path: &Path) -> Result<()> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read env file {}", path.display()))?;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        std::env::set_var(key.trim(), value);
    }
    Ok(())
}

fn read_latest_mids(path: &Path, tail_bytes: usize) -> Result<HashMap<String, f64>> {
    let value = read_latest_telemetry_row(path, tail_bytes)?;
    mids_from_row(&value)
}

fn read_latest_telemetry_row(path: &Path, tail_bytes: usize) -> Result<Value> {
    let mut file =
        File::open(path).with_context(|| format!("read telemetry {}", path.display()))?;
    let size = file
        .metadata()
        .with_context(|| format!("stat telemetry {}", path.display()))?
        .len();
    if size == 0 {
        bail!("telemetry file has no parseable rows");
    }

    let mut window = tail_bytes.max(1024).min(8 * 1024 * 1024);
    loop {
        let chunk = read_tail_chunk(&mut file, size, window)
            .with_context(|| format!("read telemetry tail {}", path.display()))?;
        if let Some(value) = parse_latest_row_from_chunk(&chunk) {
            return Ok(value);
        }
        if window as u64 >= size {
            break;
        }
        window = (window.saturating_mul(2)).min(size as usize);
    }

    bail!("telemetry file has no parseable rows");
}

fn read_tail_chunk(file: &mut File, size: u64, tail_bytes: usize) -> Result<Vec<u8>> {
    let window = (tail_bytes as u64).min(size) as usize;
    let start = size.saturating_sub(window as u64);
    file.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0_u8; window];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

fn parse_latest_row_from_chunk(chunk: &[u8]) -> Option<Value> {
    let text = String::from_utf8_lossy(chunk);
    let mut lines = text.lines();
    let mut filtered = lines.by_ref().collect::<Vec<_>>();
    if !chunk.is_empty() && chunk[0] != b'{' {
        if !filtered.is_empty() {
            filtered.remove(0);
        }
    }

    filtered
        .into_iter()
        .rev()
        .filter(|line| !line.trim().is_empty())
        .find_map(|line| serde_json::from_str::<Value>(line).ok())
}

fn mids_from_row(value: &Value) -> Result<HashMap<String, f64>> {
    let mids = value
        .get("venue_mid_usd")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("latest telemetry row missing venue_mid_usd"))?;
    let mut out = HashMap::new();
    for (idx, venue_id) in VENUE_IDS.iter().enumerate() {
        if let Some(mid) = mids.get(idx).and_then(|v| v.as_f64()) {
            out.insert((*venue_id).to_string(), mid);
        }
    }
    Ok(out)
}

fn cleanup_price(mid: f64, pos: f64, slippage_bps: f64) -> Result<(Side, f64)> {
    if !mid.is_finite() || mid <= 0.0 {
        bail!("invalid mid price {mid}");
    }
    let bump = (slippage_bps / 10_000.0).max(0.0005);
    if pos > 0.0 {
        Ok((Side::Sell, mid * (1.0 - bump)))
    } else if pos < 0.0 {
        Ok((Side::Buy, mid * (1.0 + bump)))
    } else {
        bail!("position already flat");
    }
}

fn estimated_cleanup_cost_usd(mid: f64, side: Side, price: f64, size: f64) -> f64 {
    if !mid.is_finite() || !price.is_finite() || !size.is_finite() || size <= 0.0 {
        return 0.0;
    }
    match side {
        Side::Buy => ((price - mid).max(0.0)) * size,
        Side::Sell => ((mid - price).max(0.0)) * size,
    }
}

fn snap_aggressive_price(price: f64, tick_size: f64, side: Side) -> Result<f64> {
    if !price.is_finite() || price <= 0.0 {
        bail!("invalid price {price}");
    }
    if !tick_size.is_finite() || tick_size <= 0.0 {
        bail!("invalid tick size {tick_size}");
    }
    let ticks = price / tick_size;
    let snapped_ticks = match side {
        Side::Buy => ticks.ceil(),
        Side::Sell => ticks.floor(),
    };
    Ok(snapped_ticks * tick_size)
}

fn cleanup_client_order_id(prefix: &str) -> String {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{prefix}_{now_ms}")
}

fn lighter_client_order_id() -> String {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let bounded = (now_ms % 281_474_976_710_655u128) as u64;
    bounded.to_string()
}

fn build_cleanup_summary(actions: Vec<CleanupAction>, settle_ms: u64) -> CleanupSummary {
    let mut venues_touched = Vec::new();
    for action in &actions {
        if !venues_touched.iter().any(|venue| venue == &action.venue) {
            venues_touched.push(action.venue.clone());
        }
    }
    let total_estimated_cleanup_cost_usd = actions
        .iter()
        .map(|action| action.estimated_cleanup_cost_usd)
        .sum::<f64>();
    let result = if actions.iter().any(|action| action.error.is_some()) {
        "partial"
    } else {
        "success"
    };
    CleanupSummary {
        result: result.to_string(),
        settle_ms,
        total_estimated_cleanup_cost_usd,
        venues_touched,
        actions,
    }
}

fn emit_cleanup_summary_logs(summary: &CleanupSummary) {
    for action in &summary.actions {
        match action.kind {
            CleanupActionKind::CancelAll => {
                if let Some(error) = &action.error {
                    println!(
                        "cleanup venue={} kind=cancel_all status={} error={}",
                        action.venue, action.status, error
                    );
                } else {
                    println!(
                        "cleanup venue={} kind=cancel_all status={} order_id={:?}",
                        action.venue, action.status, action.order_id
                    );
                }
            }
            CleanupActionKind::ReduceOnlyIoc => {
                println!(
                    "cleanup venue={} kind=reduce_only_ioc side={:?} size={} price={} order_id={:?} est_cost_usd={}",
                    action.venue,
                    action.side,
                    action.requested_size_base.unwrap_or(0.0),
                    action.limit_price_usd.unwrap_or(0.0),
                    action.order_id,
                    action.estimated_cleanup_cost_usd
                );
            }
        }
    }
    println!(
        "cleanup result={} venues_touched={} total_estimated_cleanup_cost_usd={} settle_ms={}",
        summary.result,
        summary.venues_touched.join(","),
        summary.total_estimated_cleanup_cost_usd,
        summary.settle_ms
    );
}

fn cancel_all_action(
    venue: &str,
    result: Result<
        paraphina::live::gateway::LiveRestResponse,
        paraphina::live::gateway::LiveGatewayError,
    >,
) -> CleanupAction {
    match result {
        Ok(resp) => CleanupAction {
            venue: venue.to_string(),
            kind: CleanupActionKind::CancelAll,
            side: None,
            requested_size_base: None,
            limit_price_usd: None,
            reference_mid_usd: None,
            slippage_bps: None,
            estimated_cleanup_cost_usd: 0.0,
            order_id: resp.order_id,
            status: "submitted".to_string(),
            error: None,
        },
        Err(err) => CleanupAction {
            venue: venue.to_string(),
            kind: CleanupActionKind::CancelAll,
            side: None,
            requested_size_base: None,
            limit_price_usd: None,
            reference_mid_usd: None,
            slippage_bps: None,
            estimated_cleanup_cost_usd: 0.0,
            order_id: None,
            status: "error".to_string(),
            error: Some(err.message),
        },
    }
}

fn reduce_only_ioc_action(
    venue: &str,
    side: Side,
    size: f64,
    price: f64,
    mid: f64,
    slippage_bps: f64,
    order_id: Option<String>,
) -> CleanupAction {
    CleanupAction {
        venue: venue.to_string(),
        kind: CleanupActionKind::ReduceOnlyIoc,
        side: Some(side),
        requested_size_base: Some(size),
        limit_price_usd: Some(price),
        reference_mid_usd: Some(mid),
        slippage_bps: Some(slippage_bps),
        estimated_cleanup_cost_usd: estimated_cleanup_cost_usd(mid, side, price, size),
        order_id,
        status: "submitted".to_string(),
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_latest_row_from_chunk_skips_partial_head() {
        let chunk = br#"partial
{"venue_mid_usd":[1,2,3,4,5]}
{"venue_mid_usd":[6,7,8,9,10]}
"#;
        let row = parse_latest_row_from_chunk(chunk).expect("latest row");
        let mids = mids_from_row(&row).expect("mids");
        assert_eq!(mids.get("extended"), Some(&6.0));
        assert_eq!(mids.get("paradex"), Some(&10.0));
    }

    #[test]
    fn read_latest_mids_recovers_from_truncated_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("telemetry.jsonl");
        let mut fh = File::create(&path).expect("create telemetry");
        writeln!(fh, "{{\"venue_mid_usd\":[1,2,3,4,5]}}").expect("write row 1");
        writeln!(fh, "{{\"venue_mid_usd\":[6,7,8,9,10]}}").expect("write row 2");
        write!(fh, "{{\"venue_mid_usd\":[11,12").expect("write truncated tail");
        drop(fh);

        let mids = read_latest_mids(&path, 32).expect("mids");
        assert_eq!(mids.get("extended"), Some(&6.0));
        assert_eq!(mids.get("paradex"), Some(&10.0));
    }

    #[test]
    fn estimated_cleanup_cost_is_conservative_for_buy_and_sell() {
        assert_eq!(
            estimated_cleanup_cost_usd(100.0, Side::Buy, 101.0, 2.0),
            2.0
        );
        assert_eq!(
            estimated_cleanup_cost_usd(100.0, Side::Sell, 99.0, 2.0),
            2.0
        );
        assert_eq!(estimated_cleanup_cost_usd(100.0, Side::Buy, 99.0, 2.0), 0.0);
        assert_eq!(
            estimated_cleanup_cost_usd(100.0, Side::Sell, 101.0, 2.0),
            0.0
        );
    }

    #[test]
    fn build_cleanup_summary_aggregates_cost_and_venues() {
        let summary = build_cleanup_summary(
            vec![
                CleanupAction {
                    venue: "aster".to_string(),
                    kind: CleanupActionKind::CancelAll,
                    side: None,
                    requested_size_base: None,
                    limit_price_usd: None,
                    reference_mid_usd: None,
                    slippage_bps: None,
                    estimated_cleanup_cost_usd: 0.0,
                    order_id: None,
                    status: "submitted".to_string(),
                    error: None,
                },
                CleanupAction {
                    venue: "aster".to_string(),
                    kind: CleanupActionKind::ReduceOnlyIoc,
                    side: Some(Side::Buy),
                    requested_size_base: Some(0.01),
                    limit_price_usd: Some(101.0),
                    reference_mid_usd: Some(100.0),
                    slippage_bps: Some(50.0),
                    estimated_cleanup_cost_usd: 0.01,
                    order_id: Some("abc".to_string()),
                    status: "submitted".to_string(),
                    error: None,
                },
            ],
            1500,
        );
        assert_eq!(summary.result, "success");
        assert_eq!(summary.settle_ms, 1500);
        assert_eq!(summary.total_estimated_cleanup_cost_usd, 0.01);
        assert_eq!(summary.venues_touched, vec!["aster".to_string()]);
    }
}

async fn cleanup_lighter(
    pos: f64,
    slippage_bps: f64,
    mids: &HashMap<String, f64>,
) -> Result<Vec<CleanupAction>> {
    let mut cfg = LighterConfig::from_env();
    cfg.venue_index = 3;
    cfg.venue_id = "lighter".to_string();
    cfg.paper_mode = false;
    let (market_tx, _market_rx) = mpsc::channel(1);
    let (exec_tx, _exec_rx) = mpsc::channel(1);
    let client = LighterConnector::new(cfg, market_tx, exec_tx);
    let mut actions = Vec::new();
    actions.push(cancel_all_action(
        "lighter",
        client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 3,
                venue_id: "lighter".to_string(),
            })
            .await,
    ));
    sleep(Duration::from_millis(500)).await;
    if pos.abs() < 1e-9 {
        return Ok(actions);
    }
    let mid = *mids
        .get("lighter")
        .ok_or_else(|| anyhow!("missing lighter mid in telemetry"))?;
    let (side, price) = cleanup_price(mid, pos, slippage_bps)?;
    let resp = client
        .place_order(LiveRestPlaceRequest {
            venue_index: 3,
            venue_id: "lighter".to_string(),
            side,
            price,
            size: pos.abs(),
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: lighter_client_order_id(),
        })
        .await
        .map_err(|err| anyhow!("lighter cleanup place_order: {}", err.message))?;
    actions.push(reduce_only_ioc_action(
        "lighter",
        side,
        pos.abs(),
        price,
        mid,
        slippage_bps,
        resp.order_id,
    ));
    Ok(actions)
}

async fn cleanup_extended(
    pos: f64,
    slippage_bps: f64,
    mids: &HashMap<String, f64>,
) -> Result<Vec<CleanupAction>> {
    let mut cfg = ExtendedConfig::from_env();
    cfg.venue_index = 0;
    let client = ExtendedRestClient::new(cfg);
    let mut actions = Vec::new();
    actions.push(cancel_all_action(
        "extended",
        client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 0,
                venue_id: "extended".to_string(),
            })
            .await,
    ));
    sleep(Duration::from_millis(500)).await;
    if pos.abs() < 1e-9 {
        return Ok(actions);
    }
    let mid = *mids
        .get("extended")
        .ok_or_else(|| anyhow!("missing extended mid in telemetry"))?;
    let (side, price) = cleanup_price(mid, pos, slippage_bps)?;
    let resp = client
        .place_order(LiveRestPlaceRequest {
            venue_index: 0,
            venue_id: "extended".to_string(),
            side,
            price,
            size: pos.abs(),
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: cleanup_client_order_id("cleanup_extended"),
        })
        .await
        .map_err(|err| anyhow!("extended cleanup place_order: {}", err.message))?;
    actions.push(reduce_only_ioc_action(
        "extended",
        side,
        pos.abs(),
        price,
        mid,
        slippage_bps,
        resp.order_id,
    ));
    Ok(actions)
}

async fn cleanup_aster(
    pos: f64,
    slippage_bps: f64,
    mids: &HashMap<String, f64>,
) -> Result<Vec<CleanupAction>> {
    let mut cfg = AsterConfig::from_env();
    cfg.venue_index = 2;
    cfg.venue_id = "aster".to_string();
    let client = AsterRestClient::new(cfg);
    let mut actions = Vec::new();
    actions.push(cancel_all_action(
        "aster",
        client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 2,
                venue_id: "aster".to_string(),
            })
            .await,
    ));
    sleep(Duration::from_millis(500)).await;
    if pos.abs() < 1e-9 {
        return Ok(actions);
    }
    let mid = *mids
        .get("aster")
        .ok_or_else(|| anyhow!("missing aster mid in telemetry"))?;
    let (side, price) = cleanup_price(mid, pos, slippage_bps)?;
    let resp = client
        .place_order(LiveRestPlaceRequest {
            venue_index: 2,
            venue_id: "aster".to_string(),
            side,
            price,
            size: pos.abs(),
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: cleanup_client_order_id("cleanup_aster"),
        })
        .await
        .map_err(|err| anyhow!("aster cleanup place_order: {}", err.message))?;
    actions.push(reduce_only_ioc_action(
        "aster",
        side,
        pos.abs(),
        price,
        mid,
        slippage_bps,
        resp.order_id,
    ));
    Ok(actions)
}

async fn cleanup_paradex(
    pos: f64,
    slippage_bps: f64,
    mids: &HashMap<String, f64>,
) -> Result<Vec<CleanupAction>> {
    let mut cfg = ParadexConfig::from_env();
    cfg.venue_index = 4;
    let client = ParadexRestClient::new(cfg);
    let mut actions = Vec::new();
    actions.push(cancel_all_action(
        "paradex",
        client
            .cancel_all(LiveRestCancelAllRequest {
                venue_index: 4,
                venue_id: "paradex".to_string(),
            })
            .await,
    ));
    sleep(Duration::from_millis(500)).await;
    if pos.abs() < 1e-9 {
        return Ok(actions);
    }
    let mid = *mids
        .get("paradex")
        .ok_or_else(|| anyhow!("missing paradex mid in telemetry"))?;
    let (side, raw_price) = cleanup_price(mid, pos, slippage_bps)?;
    let price = snap_aggressive_price(raw_price, 0.01, side)?;
    let resp = client
        .place_order(LiveRestPlaceRequest {
            venue_index: 4,
            venue_id: "paradex".to_string(),
            side,
            price,
            size: pos.abs(),
            purpose: OrderPurpose::Exit,
            time_in_force: TimeInForce::Ioc,
            post_only: false,
            reduce_only: true,
            client_order_id: cleanup_client_order_id("cleanup_paradex"),
        })
        .await
        .map_err(|err| anyhow!("paradex cleanup place_order: {}", err.message))?;
    actions.push(reduce_only_ioc_action(
        "paradex",
        side,
        pos.abs(),
        price,
        mid,
        slippage_bps,
        resp.order_id,
    ));
    Ok(actions)
}
