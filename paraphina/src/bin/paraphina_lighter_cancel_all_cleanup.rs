#[cfg(not(feature = "live_lighter"))]
fn main() {
    eprintln!(
        "paraphina_lighter_cancel_all_cleanup | status=blocked reason=live_lighter_feature_required"
    );
    std::process::exit(2);
}

#[cfg(feature = "live_lighter")]
mod cleanup {
    use paraphina::live::connectors::lighter::{LighterConfig, LighterConnector};
    use paraphina::live::gateway::{LiveRestCancelAllRequest, LiveRestClient};
    use paraphina::live::types::{ExecutionEvent, MarketDataEvent};
    use std::env;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    const APPROVAL_ENV: &str = "PARAPHINA_LIGHTER_CANCEL_ALL_CLEANUP_APPROVED";
    const REQUIRED_ENV_NAMES: &[&str] = &[
        "LIGHTER_API_KEY_INDEX",
        "LIGHTER_ACCOUNT_INDEX",
        "LIGHTER_API_PRIVATE_KEY_HEX",
        "LIGHTER_SIGNER_URL",
    ];

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
                venue_id: "lighter".to_string(),
                venue_index: 0,
            }
        }
    }

    pub async fn run() -> Result<(), String> {
        let args = parse_args(env::args().skip(1))?;
        validate_cleanup_request(&args)?;
        if args.dry_run {
            println!(
                "paraphina_lighter_cancel_all_cleanup | status=dry_run_ok venue=lighter approval_gate=set required_env_names=set"
            );
            return Ok(());
        }

        let mut cfg = LighterConfig::from_env();
        cfg.venue_index = args.venue_index;
        if cfg.paper_mode {
            return Err("LIGHTER_PAPER_MODE must be false for live cleanup".to_string());
        }
        if !cfg.has_auth() {
            return Err("required Lighter auth names are missing or invalid".to_string());
        }
        if !cfg.has_signer() {
            return Err("LIGHTER_SIGNER_URL is missing".to_string());
        }

        let (market_tx, _market_rx) = mpsc::channel::<MarketDataEvent>(1);
        let (exec_tx, _exec_rx) = mpsc::channel::<ExecutionEvent>(1);
        let connector = Arc::new(LighterConnector::new(cfg, market_tx, exec_tx));
        let req = LiveRestCancelAllRequest {
            venue_index: args.venue_index,
            venue_id: args.venue_id,
        };
        match connector.cancel_all(req).await {
            Ok(_) => {
                println!(
                    "paraphina_lighter_cancel_all_cleanup | status=sent venue=lighter action=cancel_all result=ok"
                );
                Ok(())
            }
            Err(err) => Err(format!(
                "cancel_all failed sanitized_kind={}",
                err.reason_label()
            )),
        }
    }

    fn validate_cleanup_request(args: &Args) -> Result<(), String> {
        if !truthy_env(APPROVAL_ENV) {
            return Err(format!("{APPROVAL_ENV} must be true/1/yes"));
        }
        if args.venue_id.to_ascii_lowercase() != "lighter" {
            return Err("only venue_id=lighter is supported".to_string());
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
                        "usage: paraphina_lighter_cancel_all_cleanup [--dry-run] [--venue-id lighter] [--venue-index 0]"
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
        fn parse_args_defaults_to_lighter_only() {
            let args = parse_args(Vec::<String>::new()).expect("args");
            assert_eq!(
                args,
                Args {
                    dry_run: false,
                    venue_id: "lighter".to_string(),
                    venue_index: 0
                }
            );
        }

        #[test]
        fn parse_args_accepts_dry_run_and_explicit_lighter() {
            let args = parse_args([
                "--dry-run".to_string(),
                "--venue-id".to_string(),
                "lighter".to_string(),
                "--venue-index".to_string(),
                "0".to_string(),
            ])
            .expect("args");
            assert!(args.dry_run);
            assert_eq!(args.venue_id, "lighter");
            assert_eq!(args.venue_index, 0);
        }

        #[test]
        fn parse_args_rejects_unknown() {
            assert!(parse_args(["--market".to_string(), "ETH-USD".to_string()]).is_err());
        }
    }
}

#[cfg(feature = "live_lighter")]
#[tokio::main]
async fn main() {
    if let Err(err) = cleanup::run().await {
        eprintln!("paraphina_lighter_cancel_all_cleanup | status=blocked reason={err}");
        std::process::exit(1);
    }
}
