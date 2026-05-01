//! One-shot Hyperliquid request-weight reservation tool.
//!
//! Dry-run is the default. `--execute` submits a paid `reserveRequestWeight`
//! action and requires exact USDC cost confirmation.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use k256::ecdsa::SigningKey;
use paraphina::live::connectors::hyperliquid::{
    fetch_clearinghouse_state, fetch_spot_clearinghouse_state, fetch_user_abstraction_raw,
    fetch_user_rate_limit, format_usdc_micros, reserve_request_weight_cost_micros,
    summarize_clearinghouse_state, summarize_spot_clearinghouse_state, HyperliquidConfig,
    HyperliquidConnector, HyperliquidNetwork,
};
use serde::Serialize;
use serde_json::json;
use sha3::{Digest, Keccak256};
use tokio::sync::mpsc;

const RESERVE_UNAVAILABLE_VIA_API_AGENT_REASON: &str =
    "reserve_unavailable_via_api_agent: quota target is funded main user, but reserveRequestWeight would be signed by API agent.";

#[derive(Debug, Parser)]
#[command(
    name = "hl_reserve_request_weight",
    about = "Dry-run or execute one Hyperliquid reserveRequestWeight action"
)]
struct Args {
    /// Request weight to reserve. Hyperliquid charges 0.0005 USDC per request.
    #[arg(long)]
    weight: u64,

    /// Submit the signed exchange action. Without this flag, no transaction is sent.
    #[arg(long)]
    execute: bool,

    /// Exact expected USDC cost. Required with --execute, e.g. 2.0745.
    #[arg(long)]
    confirm_cost_usdc: Option<String>,

    /// Env file(s) to load before reading HL_* settings.
    #[arg(long = "env-file")]
    env_files: Vec<PathBuf>,

    /// User address for pre/post quota and clearinghouse checks. Defaults to HL_VAULT_ADDRESS then HL_USER.
    #[arg(long)]
    user: Option<String>,

    /// Include target user as vaultAddress in the signed exchange envelope.
    #[arg(long)]
    force_vault_address: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ReserveRequestWeightPreflight {
    status: &'static str,
    failure_reason: Option<&'static str>,
    signer_address: Option<String>,
    quota_target_user: Option<String>,
    signer_user_role: Option<String>,
    signer_is_api_agent: bool,
    signer_differs_from_quota_target: Option<bool>,
}

impl ReserveRequestWeightPreflight {
    fn refused(&self) -> bool {
        self.status == "refused"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.weight == 0 {
        anyhow::bail!("--weight must be positive");
    }
    for env_file in &args.env_files {
        load_env_file(env_file)?;
    }

    let cost_micros = reserve_request_weight_cost_micros(args.weight);
    let cost_usdc = format_usdc_micros(cost_micros);
    if args.execute && args.confirm_cost_usdc.as_deref() != Some(cost_usdc.as_str()) {
        anyhow::bail!(
            "--execute requires --confirm-cost-usdc {} for weight {}",
            cost_usdc,
            args.weight
        );
    }

    let cfg = HyperliquidConfig::from_env();
    let network = match cfg.network {
        HyperliquidNetwork::Mainnet => "mainnet",
        HyperliquidNetwork::Testnet => "testnet",
    };
    let target_user = args
        .user
        .or_else(|| std::env::var("HL_VAULT_ADDRESS").ok())
        .or_else(|| std::env::var("HL_USER").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let forced_vault_address = if args.force_vault_address {
        Some(
            target_user
                .as_deref()
                .ok_or_else(|| {
                    anyhow::anyhow!("--force-vault-address requires --user or HL_VAULT_ADDRESS")
                })?
                .to_string(),
        )
    } else {
        None
    };
    let client = reqwest::Client::new();
    let signer_address = cfg
        .private_key_hex
        .as_deref()
        .map(derive_hyperliquid_signer_address)
        .transpose()?;
    let signer_user_role = if let Some(signer_address) = signer_address.as_deref() {
        fetch_user_role_label(&client, cfg.info_url(), signer_address)
            .await
            .with_context(|| format!("fetch userRole for signer {signer_address}"))?
    } else {
        None
    };
    let reserve_preflight =
        reserve_request_weight_preflight(signer_address, target_user.clone(), signer_user_role);

    let mut pre_user_rate_limit = None;
    let mut pre_user_abstraction = None;
    let mut pre_clearinghouse_state = None;
    let mut pre_spot_clearinghouse_state = None;
    if let Some(user) = target_user.as_deref() {
        pre_user_rate_limit = Some(
            fetch_user_rate_limit(&client, cfg.info_url(), user)
                .await
                .with_context(|| format!("fetch userRateLimit for {user}"))?,
        );
        pre_user_abstraction = Some(
            fetch_user_abstraction_raw(&client, cfg.info_url(), user)
                .await
                .with_context(|| format!("fetch userAbstraction for {user}"))?,
        );
        pre_clearinghouse_state = Some(summarize_clearinghouse_state(
            &fetch_clearinghouse_state(&client, cfg.info_url(), user)
                .await
                .with_context(|| format!("fetch clearinghouseState for {user}"))?,
        ));
        pre_spot_clearinghouse_state = Some(summarize_spot_clearinghouse_state(
            &fetch_spot_clearinghouse_state(&client, cfg.info_url(), user)
                .await
                .with_context(|| format!("fetch spotClearinghouseState for {user}"))?,
        ));
    }

    let mut exchange_response = None;
    let mut post_user_rate_limit = None;
    let reserve_refused = args.execute && reserve_preflight.refused();
    if args.execute && !reserve_refused {
        if cfg
            .private_key_hex
            .as_deref()
            .map(str::trim)
            .unwrap_or_default()
            .is_empty()
        {
            anyhow::bail!("HL_PRIVATE_KEY is required for --execute");
        }
        let (market_tx, _market_rx) = mpsc::channel(1);
        let (exec_tx, _exec_rx) = mpsc::channel(1);
        let connector = HyperliquidConnector::new(cfg.clone(), market_tx, exec_tx);
        exchange_response = Some(
            if let Some(vault_address) = forced_vault_address.as_deref() {
                connector
                    .reserve_request_weight_for_vault_address(args.weight, vault_address)
                    .await?
            } else {
                connector.reserve_request_weight(args.weight).await?
            },
        );
        if let Some(user) = target_user.as_deref() {
            post_user_rate_limit = Some(
                fetch_user_rate_limit(&client, cfg.info_url(), user)
                    .await
                    .with_context(|| format!("fetch post userRateLimit for {user}"))?,
            );
        }
    }

    let report = json!({
        "schema_version": 1,
        "status": if reserve_refused { "refused" } else if args.execute { "executed" } else { "dry_run" },
        "network": network,
        "rest_url": cfg.rest_url(),
        "info_url": cfg.info_url(),
        "target_user": target_user,
        "forced_vault_address": forced_vault_address,
        "reserve_request_weight_preflight": reserve_preflight,
        "weight": args.weight,
        "cost_usdc": cost_usdc,
        "execute": args.execute,
        "submitted": args.execute && !reserve_refused,
        "pre_user_rate_limit": pre_user_rate_limit,
        "pre_user_abstraction": pre_user_abstraction,
        "pre_clearinghouse_state": pre_clearinghouse_state,
        "pre_spot_clearinghouse_state": pre_spot_clearinghouse_state,
        "exchange_response": exchange_response,
        "post_user_rate_limit": post_user_rate_limit,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if reserve_refused {
        anyhow::bail!(RESERVE_UNAVAILABLE_VIA_API_AGENT_REASON);
    }
    Ok(())
}

fn reserve_request_weight_preflight(
    signer_address: Option<String>,
    quota_target_user: Option<String>,
    signer_user_role: Option<String>,
) -> ReserveRequestWeightPreflight {
    let signer_is_api_agent = signer_user_role
        .as_deref()
        .map(|role| role.eq_ignore_ascii_case("agent"))
        .unwrap_or(false);
    let signer_differs_from_quota_target = match (&signer_address, &quota_target_user) {
        (Some(signer), Some(target)) => Some(!same_hl_address(signer, target)),
        _ => None,
    };
    let refused = signer_is_api_agent && signer_differs_from_quota_target == Some(true);
    ReserveRequestWeightPreflight {
        status: if refused { "refused" } else { "pass" },
        failure_reason: refused.then_some(RESERVE_UNAVAILABLE_VIA_API_AGENT_REASON),
        signer_address,
        quota_target_user,
        signer_user_role,
        signer_is_api_agent,
        signer_differs_from_quota_target,
    }
}

fn same_hl_address(left: &str, right: &str) -> bool {
    left.trim().eq_ignore_ascii_case(right.trim())
}

fn derive_hyperliquid_signer_address(private_key_hex: &str) -> Result<String> {
    let key_hex = private_key_hex
        .trim()
        .strip_prefix("0x")
        .unwrap_or_else(|| private_key_hex.trim());
    let key_bytes = hex::decode(key_hex).context("decode HL_PRIVATE_KEY hex")?;
    let signing_key =
        SigningKey::from_slice(&key_bytes).context("parse HL_PRIVATE_KEY as secp256k1 key")?;
    let encoded = signing_key.verifying_key().to_encoded_point(false);
    let digest = Keccak256::digest(&encoded.as_bytes()[1..]);
    Ok(format!("0x{}", hex::encode(&digest[12..])))
}

async fn fetch_user_role_label(
    client: &reqwest::Client,
    info_url: &str,
    user: &str,
) -> Result<Option<String>> {
    let response = client
        .post(info_url)
        .json(&json!({
            "type": "userRole",
            "user": user,
        }))
        .send()
        .await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        anyhow::bail!("http {}: {}", status, body);
    }
    let value: serde_json::Value = serde_json::from_str(&body)?;
    Ok(value
        .get("role")
        .and_then(|role| role.as_str())
        .map(str::to_string))
}

fn load_env_file(path: &Path) -> Result<()> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("read env file {}", path.display()))?;
    for (line_no, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let trimmed = trimmed.strip_prefix("export ").unwrap_or(trimmed);
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        {
            anyhow::bail!("invalid env key at {}:{}", path.display(), line_no + 1);
        }
        std::env::set_var(key, strip_env_quotes(value.trim()));
    }
    Ok(())
}

fn strip_env_quotes(value: &str) -> String {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        if (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'')
        {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_preflight_refuses_api_agent_target_mismatch() {
        let preflight = reserve_request_weight_preflight(
            Some("0x00000000000000000000000000000000000000aa".to_string()),
            Some("0x00000000000000000000000000000000000000bb".to_string()),
            Some("agent".to_string()),
        );

        assert!(preflight.refused());
        assert_eq!(
            preflight.failure_reason,
            Some(RESERVE_UNAVAILABLE_VIA_API_AGENT_REASON)
        );
        assert!(preflight.signer_is_api_agent);
        assert_eq!(preflight.signer_differs_from_quota_target, Some(true));
    }

    #[test]
    fn reserve_preflight_allows_same_api_agent_target() {
        let preflight = reserve_request_weight_preflight(
            Some("0x00000000000000000000000000000000000000AA".to_string()),
            Some("0x00000000000000000000000000000000000000aa".to_string()),
            Some("agent".to_string()),
        );

        assert!(!preflight.refused());
        assert_eq!(preflight.status, "pass");
        assert_eq!(preflight.failure_reason, None);
        assert_eq!(preflight.signer_differs_from_quota_target, Some(false));
    }

    #[test]
    fn reserve_preflight_allows_non_agent_target_mismatch() {
        let preflight = reserve_request_weight_preflight(
            Some("0x00000000000000000000000000000000000000aa".to_string()),
            Some("0x00000000000000000000000000000000000000bb".to_string()),
            Some("user".to_string()),
        );

        assert!(!preflight.refused());
        assert_eq!(preflight.status, "pass");
        assert!(!preflight.signer_is_api_agent);
        assert_eq!(preflight.signer_differs_from_quota_target, Some(true));
    }

    #[test]
    fn derives_signer_address_from_private_key_without_printing_secret() {
        let address = derive_hyperliquid_signer_address(
            "0x0000000000000000000000000000000000000000000000000000000000000001",
        )
        .expect("derive signer");

        assert_eq!(address, "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf");
    }
}
