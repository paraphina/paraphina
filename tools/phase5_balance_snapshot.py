#!/usr/bin/env python3
"""Capture exact Phase 5 all-venue account balances."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from phase5_econ_attribution import (
    DEFAULT_ENV_FILE,
    ASTER_DOCS,
    EXTENDED_DOCS,
    HYPERLIQUID_DOCS,
    LIGHTER_DOCS,
    PARADEX_DOCS,
    decimal_of,
    http_json,
    lighter_auth_token,
    load_env,
    load_baseline_manifest,
    paradex_token,
    sign_aster_query,
    utc_now_iso,
)


ZERO = Decimal("0")
VENUE_ORDER = ("hyperliquid", "extended", "lighter", "aster", "paradex")
UTC = timezone.utc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def decimal_str(value: Any) -> str:
    return format(decimal_of(value), "f")


def parse_utc_to_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def default_window(captured_at_utc: str) -> tuple[str, str]:
    end_dt = datetime.fromisoformat(captured_at_utc.replace("Z", "+00:00"))
    start_dt = end_dt - timedelta(days=1)
    return (
        start_dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        captured_at_utc,
    )


def raw_path(root: Path, label: str, venue: str) -> Path:
    return root / "balances" / f"{label}_raw" / f"{venue}.json"


def details_path(root: Path, label: str, venue: str) -> Path:
    return root / "balances" / f"{label}_details" / f"{venue}.json"


def hyperliquid_spot_usdc(spot_state: dict[str, Any]) -> Decimal:
    for balance in spot_state.get("balances") or []:
        if str(balance.get("coin") or "").upper() == "USDC":
            return decimal_of(balance.get("total"))
    return ZERO


def hyperliquid_perps_account_value(clearinghouse_state: dict[str, Any]) -> Decimal:
    margin_summary = clearinghouse_state.get("marginSummary")
    if not isinstance(margin_summary, dict):
        return ZERO
    return decimal_of(margin_summary.get("accountValue"))


def hyperliquid_uses_separate_perps_balance(account_mode: str) -> bool:
    return account_mode in {"standard", "disabled"}


def row_payload(
    *,
    venue: str,
    captured_at_utc: str,
    balance_usd: Decimal,
    balance_components: dict[str, str],
    account_ref: dict[str, Any],
    raw_file: Path,
    details_file: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "venue": venue,
        "captured_at_utc": captured_at_utc,
        "balance_usd": decimal_str(balance_usd),
        "balance_components": balance_components,
        "account_ref": account_ref,
        "raw_path": str(raw_file),
        "details_path": str(details_file),
        "source": source,
    }


def collect_hyperliquid(env: dict[str, str], out_dir: Path, label: str, captured_at_utc: str) -> dict[str, Any]:
    info_url = env.get("HL_INFO_URL", "").strip() or "https://api.hyperliquid.xyz/info"
    user = env["HL_VAULT_ADDRESS"]
    account_mode = http_json("POST", info_url, payload={"type": "userAbstraction", "user": user})
    clearinghouse_state = http_json("POST", info_url, payload={"type": "clearinghouseState", "user": user})
    spot_state = http_json("POST", info_url, payload={"type": "spotClearinghouseState", "user": user})
    spot_usdc = hyperliquid_spot_usdc(spot_state)
    perps_account_value = hyperliquid_perps_account_value(clearinghouse_state)
    account_mode_label = str(account_mode)
    if hyperliquid_uses_separate_perps_balance(account_mode_label):
        current_total = perps_account_value + spot_usdc
        selector = "clearinghouseState.marginSummary.accountValue + spotClearinghouseState.USDC.total"
    else:
        current_total = spot_usdc
        selector = "spotClearinghouseState.USDC.total"
    raw_file = raw_path(out_dir, label, "hyperliquid")
    details_file = details_path(out_dir, label, "hyperliquid")
    raw = {
        "userAbstraction": account_mode,
        "clearinghouseState": clearinghouse_state,
        "spotClearinghouseState": spot_state,
    }
    row = row_payload(
        venue="hyperliquid",
        captured_at_utc=captured_at_utc,
        balance_usd=current_total,
        balance_components={
            "account_mode": account_mode_label,
            "perps_account_value_usd": decimal_str(perps_account_value),
            "spot_usdc_total": decimal_str(spot_usdc),
            "total_usd": decimal_str(current_total),
        },
        account_ref={"user": user},
        raw_file=raw_file,
        details_file=details_file,
        source={"endpoint": info_url, "selector": selector, "docs": HYPERLIQUID_DOCS},
    )
    write_json(raw_file, raw)
    write_json(details_file, row)
    return row


def collect_extended(env: dict[str, str], out_dir: Path, label: str, captured_at_utc: str) -> dict[str, Any]:
    base_url = env.get("EXTENDED_REST_URL", "https://api.starknet.extended.exchange").rstrip("/")
    headers = {"X-Api-Key": env["EXTENDED_API_KEY"]}
    account_info = http_json("GET", f"{base_url}/api/v1/user/account/info", headers=headers)
    account_id = int((account_info.get("data") or {}).get("accountId"))
    equities = http_json(
        "GET",
        f"{base_url}/api/v1/portfolio/charts/equities",
        headers=headers,
        params={"accountId": account_id, "interval": "ALL"},
    )
    equity_rows = [item for item in equities.get("data") or [] if isinstance(item, dict)]
    if not equity_rows:
        raise RuntimeError("extended equity chart returned no rows")
    current_balance = decimal_of(equity_rows[-1]["value"])
    raw_file = raw_path(out_dir, label, "extended")
    details_file = details_path(out_dir, label, "extended")
    raw = {"account_info": account_info, "equities": equities}
    row = row_payload(
        venue="extended",
        captured_at_utc=captured_at_utc,
        balance_usd=current_balance,
        balance_components={"equity_value_usd": decimal_str(current_balance)},
        account_ref={"account_id": account_id},
        raw_file=raw_file,
        details_file=details_file,
        source={
            "endpoint": f"{base_url}/api/v1/portfolio/charts/equities",
            "selector": "data[-1].value",
            "docs": EXTENDED_DOCS,
        },
    )
    write_json(raw_file, raw)
    write_json(details_file, row)
    return row


def collect_lighter(
    env: dict[str, str],
    out_dir: Path,
    label: str,
    captured_at_utc: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    network = env.get("LIGHTER_NETWORK", "mainnet").strip().lower()
    base_url = (
        env.get("LIGHTER_HTTP_BASE_URL", "").strip()
        or env.get("LIGHTER_REST_URL", "").strip()
        or ("https://testnet.zklighter.elliot.ai" if network == "testnet" else "https://mainnet.zklighter.elliot.ai")
    ).rstrip("/")
    account_index = int(env["LIGHTER_ACCOUNT_INDEX"])
    auth_token = lighter_auth_token(env)
    account_payload = http_json(
        "GET",
        f"{base_url}/api/v1/account",
        params={"by": "index", "value": account_index},
    )
    account = ((account_payload.get("accounts") or [account_payload])[0]) if isinstance(account_payload, dict) else account_payload
    pnl_payload = http_json(
        "GET",
        f"{base_url}/api/v1/pnl",
        headers={"Authorization": auth_token},
        params={
            "by": "index",
            "value": account_index,
            "resolution": "1d",
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "count_back": 64,
            "ignore_transfers": "true",
        },
    )
    pnl_rows = [item for item in pnl_payload.get("pnl") or [] if isinstance(item, dict)]
    if not pnl_rows:
        raise RuntimeError("lighter pnl endpoint returned no rows")
    last_row = pnl_rows[-1]
    current_collateral = decimal_of(account.get("collateral"))
    current_spot_value = decimal_of(last_row.get("trade_spot_pnl"))
    current_total = current_collateral + current_spot_value
    l1_address = str(account.get("l1_address") or account.get("l1Address") or "")
    raw_file = raw_path(out_dir, label, "lighter")
    details_file = details_path(out_dir, label, "lighter")
    raw = {"account": account_payload, "pnl": pnl_payload}
    row = row_payload(
        venue="lighter",
        captured_at_utc=captured_at_utc,
        balance_usd=current_total,
        balance_components={
            "perps_usd": decimal_str(current_collateral),
            "spot_usd": decimal_str(current_spot_value),
            "total_usd": decimal_str(current_total),
        },
        account_ref={"account_index": account_index, "l1_address": l1_address},
        raw_file=raw_file,
        details_file=details_file,
        source={
            "account_endpoint": f"{base_url}/api/v1/account",
            "pnl_endpoint": f"{base_url}/api/v1/pnl",
            "selector": "account.collateral + pnl[-1].trade_spot_pnl",
            "docs": LIGHTER_DOCS,
        },
    )
    write_json(raw_file, raw)
    write_json(details_file, row)
    return row


def collect_aster(env: dict[str, str], out_dir: Path, label: str, captured_at_utc: str) -> dict[str, Any]:
    rest_url = env.get("ASTER_REST_URL", "https://fapi.asterdex.com").rstrip("/")
    api_key = env["ASTER_API_KEY"]
    account = http_json(
        "GET",
        f"{rest_url}/fapi/v2/account?{sign_aster_query(env, {})}",
        headers={"X-MBX-APIKEY": api_key},
    )
    current_balance = decimal_of(account.get("totalWalletBalance"))
    raw_file = raw_path(out_dir, label, "aster")
    details_file = details_path(out_dir, label, "aster")
    row = row_payload(
        venue="aster",
        captured_at_utc=captured_at_utc,
        balance_usd=current_balance,
        balance_components={"total_wallet_balance_usd": decimal_str(current_balance)},
        account_ref={"assets_count": len(account.get("assets") or [])},
        raw_file=raw_file,
        details_file=details_file,
        source={"endpoint": f"{rest_url}/fapi/v2/account", "selector": "totalWalletBalance", "docs": ASTER_DOCS},
    )
    write_json(raw_file, account)
    write_json(details_file, row)
    return row


def collect_paradex(env: dict[str, str], out_dir: Path, label: str, captured_at_utc: str) -> dict[str, Any]:
    rest_url = env.get("PARADEX_REST_URL", "https://api.prod.paradex.trade/v1").rstrip("/")
    token = paradex_token(env)
    account = http_json(
        "GET",
        f"{rest_url}/account",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    current_balance = decimal_of(account.get("account_value"))
    raw_file = raw_path(out_dir, label, "paradex")
    details_file = details_path(out_dir, label, "paradex")
    row = row_payload(
        venue="paradex",
        captured_at_utc=captured_at_utc,
        balance_usd=current_balance,
        balance_components={"account_value_usd": decimal_str(current_balance)},
        account_ref={"account": account.get("account")},
        raw_file=raw_file,
        details_file=details_file,
        source={"endpoint": f"{rest_url}/account", "selector": "account_value", "docs": PARADEX_DOCS},
    )
    write_json(raw_file, account)
    write_json(details_file, row)
    return row


def collect_rows(
    env: dict[str, str],
    out_dir: Path,
    label: str,
    captured_at_utc: str,
    start_ms: int,
    end_ms: int,
) -> list[dict[str, Any]]:
    rows = [
        collect_hyperliquid(env, out_dir, label, captured_at_utc),
        collect_extended(env, out_dir, label, captured_at_utc),
        collect_lighter(env, out_dir, label, captured_at_utc, start_ms, end_ms),
        collect_aster(env, out_dir, label, captured_at_utc),
        collect_paradex(env, out_dir, label, captured_at_utc),
    ]
    return sorted(rows, key=lambda row: VENUE_ORDER.index(row["venue"]))


def manifest_from_rows(rows: list[dict[str, Any]], captured_at_utc: str, label: str) -> dict[str, Any]:
    by_venue = {row["venue"]: row for row in rows}
    lighter = by_venue["lighter"]["balance_components"]
    return {
        "captured_at_utc": captured_at_utc,
        "source": f"phase5_balance_snapshot:{label}",
        "lighter_spot_included": True,
        "venues": {
            "hyperliquid": {"balance_usd": by_venue["hyperliquid"]["balance_usd"]},
            "extended": {"balance_usd": by_venue["extended"]["balance_usd"]},
            "lighter": {
                "perps_usd": lighter["perps_usd"],
                "spot_usd": lighter["spot_usd"],
                "total_usd": lighter["total_usd"],
            },
            "aster": {"balance_usd": by_venue["aster"]["balance_usd"]},
            "paradex": {"balance_usd": by_venue["paradex"]["balance_usd"]},
        },
    }


def total_balance_usd(rows: list[dict[str, Any]]) -> Decimal:
    return sum((decimal_of(row["balance_usd"]) for row in rows), ZERO)


def load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def compare_to_pre(pre_snapshot_path: Path, post_rows: list[dict[str, Any]], post_snapshot_path: Path) -> dict[str, Any]:
    pre = load_snapshot(pre_snapshot_path)
    pre_rows = {row["venue"]: row for row in pre.get("rows", []) if isinstance(row, dict)}
    post_by_venue = {row["venue"]: row for row in post_rows}
    per_venue: dict[str, Any] = {}
    for venue in VENUE_ORDER:
        pre_balance = decimal_of(pre_rows[venue]["balance_usd"])
        post_balance = decimal_of(post_by_venue[venue]["balance_usd"])
        per_venue[venue] = {
            "pre_balance_usd": decimal_str(pre_balance),
            "post_balance_usd": decimal_str(post_balance),
            "delta_usd": decimal_str(post_balance - pre_balance),
        }
    pre_total = decimal_of(pre["total_balance_usd"])
    post_total = total_balance_usd(post_rows)
    total_delta = post_total - pre_total
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "pre_snapshot_path": str(pre_snapshot_path),
        "post_snapshot_path": str(post_snapshot_path),
        "venue_count": len(per_venue),
        "venues": list(VENUE_ORDER),
        "total": {
            "pre_usd": decimal_str(pre_total),
            "post_usd": decimal_str(post_total),
            "delta_usd": decimal_str(total_delta),
            "abs_delta_usd": decimal_str(abs(total_delta)),
            "abs_delta_usd_float": float(abs(total_delta)),
        },
        "per_venue": per_venue,
    }


def build_snapshot(
    *,
    label: str,
    env_file: Path,
    out_dir: Path,
    start_utc: str | None,
    end_utc: str | None,
    pre_snapshot: Path | None,
) -> dict[str, Any]:
    captured_at_utc = utc_now_iso()
    default_start_utc, default_end_utc = default_window(captured_at_utc)
    window_start_utc = start_utc or default_start_utc
    window_end_utc = end_utc or default_end_utc
    start_ms = parse_utc_to_ms(window_start_utc)
    end_ms = parse_utc_to_ms(window_end_utc)
    if end_ms <= start_ms:
        raise ValueError("--end-utc must be after --start-utc")

    out_dir.mkdir(parents=True, exist_ok=True)
    env = load_env(env_file)
    rows = collect_rows(env, out_dir, label, captured_at_utc, start_ms, end_ms)
    manifest = manifest_from_rows(rows, captured_at_utc, label)
    snapshot_path = out_dir / f"balance_{label}_snapshot.json"
    manifest_path = out_dir / f"balance_{label}_manifest.yaml"
    total = total_balance_usd(rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "captured_at_utc": captured_at_utc,
        "env_file": str(env_file),
        "collection_window": {
            "start_utc": window_start_utc,
            "end_utc": window_end_utc,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
        "venue_count": len(rows),
        "venues": list(VENUE_ORDER),
        "total_balance_usd": decimal_str(total),
        "rows": rows,
        "baseline_manifest_path": str(manifest_path),
    }
    if pre_snapshot is not None:
        payload["comparison_to_pre"] = compare_to_pre(pre_snapshot, rows, snapshot_path)
    write_json(snapshot_path, payload)
    write_yaml(manifest_path, manifest)
    if label == "pre":
        write_yaml(out_dir / "balance_pre_manifest.yaml", manifest)
    if label == "post" and "comparison_to_pre" in payload:
        write_json(out_dir / "balance_snapshot_comparison.json", payload["comparison_to_pre"])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture exact Phase 5 all-venue account balances")
    parser.add_argument("--label", required=True, choices=("pre", "post"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--pre-snapshot", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_snapshot(
        label=args.label,
        env_file=args.env_file,
        out_dir=args.out_dir,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        pre_snapshot=args.pre_snapshot,
    )
    print(json.dumps({"snapshot": f"balance_{args.label}_snapshot.json", "total_balance_usd": payload["total_balance_usd"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
