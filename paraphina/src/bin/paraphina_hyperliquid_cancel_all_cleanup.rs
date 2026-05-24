#[cfg(not(feature = "live_hyperliquid"))]
fn main() {
    eprintln!(
        "paraphina_hyperliquid_cancel_all_cleanup | status=blocked reason=live_hyperliquid_feature_required"
    );
    std::process::exit(2);
}

#[cfg(feature = "live_hyperliquid")]
mod cleanup {
    use paraphina::live::connectors::hyperliquid::{HyperliquidConfig, HyperliquidConnector};
    use paraphina::live::gateway::{LiveRestCancelRequest, LiveRestClient, TransportHint};
    use paraphina::live::types::{ExecutionEvent, MarketDataEvent};
    use serde_json::{json, Value};
    use std::env;
    use tokio::sync::mpsc;

    const APPROVAL_ENV: &str = "PARAPHINA_HYPERLIQUID_CANCEL_ALL_CLEANUP_APPROVED";
    const REQUIRED_ENV_NAMES: &[&str] = &["HL_PRIVATE_KEY", "HL_VAULT_ADDRESS"];

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Args {
        dry_run: bool,
        venue_id: String,
        venue_index: usize,
    }

    impl Default for Args {
        fn default() -> Self {
            Self {
                dry_run: false,
                venue_id: "hyperliquid".to_string(),
                venue_index: 0,
            }
        }
    }

    pub async fn run() -> Result<(), String> {
        let args = parse_args(env::args().skip(1))?;
        validate_cleanup_request(&args)?;
        if args.dry_run {
            println!(
                "paraphina_hyperliquid_cancel_all_cleanup | status=dry_run_ok venue=hyperliquid approval_gate=set required_env_names=set"
            );
            return Ok(());
        }

        let mut cfg = HyperliquidConfig::from_env();
        cfg.venue_index = args.venue_index;
        if cfg.paper_mode {
            return Err("HL_PAPER_MODE must be false for live cleanup".to_string());
        }
        if cfg
            .private_key_hex
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
            || cfg.vault_address.as_deref().unwrap_or("").trim().is_empty()
        {
            return Err("required Hyperliquid auth names are missing".to_string());
        }
        let request_cfg = cfg.clone();

        let (market_tx, _market_rx) = mpsc::channel::<MarketDataEvent>(1);
        let (exec_tx, _exec_rx) = mpsc::channel::<ExecutionEvent>(1);
        let connector = HyperliquidConnector::new(cfg, market_tx, exec_tx);
        let cancel_reqs = open_order_cancel_requests(&request_cfg, &args).await?;
        if cancel_reqs.is_empty() {
            println!(
                "paraphina_hyperliquid_cancel_all_cleanup | status=clean venue=hyperliquid action=cancel_explicit open_order_count=0"
            );
            return Ok(());
        }
        let attempted = cancel_reqs.len();
        let results = connector
            .cancel_batch_with_hint(cancel_reqs, TransportHint::HyperliquidSyncControl)
            .await;
        let failed = results.iter().filter(|result| result.is_err()).count();
        if failed == 0 {
            println!(
                "paraphina_hyperliquid_cancel_all_cleanup | status=sent venue=hyperliquid action=cancel_explicit result=ok attempted_count={}",
                attempted
            );
            Ok(())
        } else {
            let labels = results
                .iter()
                .filter_map(|result| result.as_ref().err().map(|err| err.reason_label()))
                .collect::<Vec<_>>()
                .join(",");
            Err(format!(
                "cancel_explicit failed attempted_count={} failed_count={} sanitized_reasons={}",
                attempted, failed, labels
            ))
        }
    }

    async fn open_order_cancel_requests(
        cfg: &HyperliquidConfig,
        args: &Args,
    ) -> Result<Vec<LiveRestCancelRequest>, String> {
        let user = cfg
            .vault_address
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "required Hyperliquid account name is missing".to_string())?;
        let client = reqwest::Client::new();
        let mut last_error = None;
        for query_type in ["openOrders", "frontendOpenOrders"] {
            let response = match client
                .post(cfg.info_url())
                .json(&json!({"type": query_type, "user": user}))
                .send()
                .await
            {
                Ok(response) => response,
                Err(_err) => {
                    last_error = Some("open_order_query_transport");
                    continue;
                }
            };
            if !response.status().is_success() {
                last_error = Some("open_order_query_status");
                continue;
            }
            let payload = match response.json::<Value>().await {
                Ok(payload) => payload,
                Err(_err) => {
                    last_error = Some("open_order_query_json");
                    continue;
                }
            };
            let orders = if let Some(array) = payload.as_array() {
                array.as_slice()
            } else {
                payload
                    .get("orders")
                    .and_then(|orders| orders.as_array())
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
            };
            let mut requests = Vec::new();
            for order in orders {
                if !order_matches_coin(order, &cfg.coin) {
                    continue;
                }
                let Some(order_id) = sanitized_order_id(order) else {
                    return Err("open_order_missing_cancel_identifier".to_string());
                };
                requests.push(LiveRestCancelRequest {
                    venue_index: args.venue_index,
                    venue_id: args.venue_id.clone(),
                    order_id,
                });
            }
            return Ok(requests);
        }
        Err(format!(
            "open_order_query_failed sanitized_reason={}",
            last_error.unwrap_or("unknown")
        ))
    }

    fn order_matches_coin(order: &Value, coin: &str) -> bool {
        let configured = coin.trim().to_ascii_uppercase();
        let observed = order
            .get("coin")
            .or_else(|| order.get("symbol"))
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .trim()
            .to_ascii_uppercase();
        observed == configured
    }

    fn sanitized_order_id(order: &Value) -> Option<String> {
        for key in ["oid", "orderId", "order_id"] {
            if let Some(value) = order.get(key) {
                if let Some(text) = value.as_str() {
                    if !text.trim().is_empty() {
                        return Some(text.trim().to_string());
                    }
                }
                if let Some(number) = value.as_u64() {
                    return Some(number.to_string());
                }
            }
        }
        None
    }

    fn validate_cleanup_request(args: &Args) -> Result<(), String> {
        if !truthy_env(APPROVAL_ENV) {
            return Err(format!("{APPROVAL_ENV} must be true/1/yes"));
        }
        if args.venue_id.to_ascii_lowercase() != "hyperliquid" {
            return Err("only venue_id=hyperliquid is supported".to_string());
        }
        if args.venue_index != 0 {
            return Err("only venue_index=0 is supported for the cleanup guard".to_string());
        }
        let missing = REQUIRED_ENV_NAMES
            .iter()
            .copied()
            .filter(|name| !env_name_is_set(name))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(format!("missing required env names: {}", missing.join(",")));
        }
        Ok(())
    }

    fn env_name_is_set(name: &str) -> bool {
        env::var(name)
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
    }

    fn truthy_env(name: &str) -> bool {
        env::var(name)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "true" | "1" | "yes"
                )
            })
            .unwrap_or(false)
    }

    fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
        let mut parsed = Args::default();
        let mut iter = args.into_iter();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--dry-run" => parsed.dry_run = true,
                "--venue-id" => {
                    parsed.venue_id = iter
                        .next()
                        .ok_or_else(|| "--venue-id requires a value".to_string())?;
                }
                "--venue-index" => {
                    let raw = iter
                        .next()
                        .ok_or_else(|| "--venue-index requires a value".to_string())?;
                    parsed.venue_index = raw
                        .parse::<usize>()
                        .map_err(|_| "--venue-index must be a non-negative integer".to_string())?;
                }
                "--help" | "-h" => {
                    return Err(
                        "usage: paraphina_hyperliquid_cancel_all_cleanup [--dry-run] [--venue-id hyperliquid] [--venue-index 0]"
                            .to_string(),
                    );
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }
        Ok(parsed)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_args_defaults_to_hyperliquid_only() {
            let args = parse_args(Vec::<String>::new()).expect("args");
            assert_eq!(
                args,
                Args {
                    dry_run: false,
                    venue_id: "hyperliquid".to_string(),
                    venue_index: 0
                }
            );
        }

        #[test]
        fn parse_args_accepts_dry_run_and_explicit_hyperliquid() {
            let args = parse_args([
                "--dry-run".to_string(),
                "--venue-id".to_string(),
                "hyperliquid".to_string(),
                "--venue-index".to_string(),
                "0".to_string(),
            ])
            .expect("args");
            assert!(args.dry_run);
            assert_eq!(args.venue_id, "hyperliquid");
            assert_eq!(args.venue_index, 0);
        }

        #[test]
        fn parse_args_rejects_unknown() {
            assert!(parse_args(["--market".to_string(), "ETH".to_string()]).is_err());
        }
    }
}

#[cfg(feature = "live_hyperliquid")]
#[tokio::main]
async fn main() {
    if let Err(err) = cleanup::run().await {
        eprintln!("paraphina_hyperliquid_cancel_all_cleanup | status=blocked reason={err}");
        std::process::exit(1);
    }
}
