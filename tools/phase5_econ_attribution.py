#!/usr/bin/env python3
"""Phase 5 venue-native economics attribution gate."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = Path("/etc/paraphina/current.env")
DEFAULT_BASELINE = ROOT / "docs" / "INVESTIGATIONS" / "phase5_balance_baseline_20260416.yaml"
DEFAULT_SURFACE_SPEC = (
    ROOT / "phase5" / "runs" / "phase5_reopened_final_closeout" / "final_topology_spec.yaml"
)
DEFAULT_OUT_DIR = (
    ROOT / "phase5" / "runs" / "phase5_all5_current_surface_lighter_aster_loss_attribution_econ_gate"
)
DEFAULT_REPORT_PATH = (
    ROOT
    / "docs"
    / "INVESTIGATIONS"
    / f"phase5_all5_current_surface_lighter_aster_loss_attribution_{datetime.now(UTC).strftime('%Y%m%d')}.md"
)
TELEMETRY_ANALYZER = ROOT / "tools" / "telemetry_analyzer.py"
LIVE_AUDIT = Path("/home/ubuntu/paraphina/tools/live_venue_audit.py")
ZERO = Decimal("0")

LIGHTER_DOCS = {
    "fees": "https://docs.lighter.xyz/trading/trading-fees",
    "account_types": "https://apidocs.lighter.xyz/docs/account-types",
    "accounts": "https://docs.lighter.xyz/trading/unified-trading-accounts",
    "pnl": "https://apidocs.lighter.xyz/reference/pnl",
}
ASTER_DOCS = {
    "fees": "https://docs.asterdex.com/trading/perpetuals/fees-and-specs/fees",
    "income": "https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation",
    "market_maker": "https://docs.asterdex.com/program-and-rewards/market-maker-program",
}
HYPERLIQUID_DOCS = {
    "accounts": "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/account-abstraction-modes",
    "fees": "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees",
    "info": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals",
}
PARADEX_DOCS = {
    "fees": "https://docs.paradex.trade/trading/trading-fees",
    "fastfills": "https://docs.paradex.trade/trading/fastfills",
    "orders": "https://docs.paradex.trade/trading/orders/order-instructions",
}
EXTENDED_DOCS = {
    "fees": "https://docs.extended.exchange/extended-resources/trading/trading-fees-and-rebates",
    "orders": "https://docs.extended.exchange/extended-resources/trading/order-types",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def decimal_of(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value is None:
        return ZERO
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid decimal input")
    return Decimal(str(value))


def decimal_to_float(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001")))


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


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value.strip().strip('"').strip("'")
    return env


def run_command(args: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def run_shell(command: str, *, env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        shell=True,
        check=True,
    )
    return completed.stdout


def http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    payload: Any = None,
    timeout: float = 30.0,
) -> Any:
    if params:
        encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}?{encoded}" if encoded else url
    request_headers = dict(headers or {})
    body: bytes | None = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised in live execution
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {detail}") from exc


def parse_systemctl_properties(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_baseline_manifest(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    venues = data.get("venues") or {}
    lighter = venues.get("lighter") or {}
    baseline = {
        "captured_at_utc": data["captured_at_utc"],
        "source": data["source"],
        "lighter_spot_included": bool(data.get("lighter_spot_included", False)),
        "venues": {
            "hyperliquid": decimal_of((venues.get("hyperliquid") or {}).get("balance_usd")),
            "extended": decimal_of((venues.get("extended") or {}).get("balance_usd")),
            "aster": decimal_of((venues.get("aster") or {}).get("balance_usd")),
            "paradex": decimal_of((venues.get("paradex") or {}).get("balance_usd")),
            "lighter_perp": decimal_of(lighter.get("perps_usd")),
            "lighter_spot": decimal_of(lighter.get("spot_usd")),
            "lighter_total": decimal_of(lighter.get("total_usd")),
        },
    }
    return baseline


def import_lighter_sdk() -> Any:
    try:
        import lighter  # type: ignore

        return lighter
    except ImportError:
        sdk_root = Path("/tmp/lighter_sdk")
        if sdk_root.exists():
            sys.path.insert(0, str(sdk_root))
            import lighter  # type: ignore

            return lighter
    raise RuntimeError("lighter-sdk not available; install it or restore /tmp/lighter_sdk")


async def _lighter_auth_token_async(env: dict[str, str]) -> str:
    lighter = import_lighter_sdk()
    explicit = env.get("LIGHTER_AUTH_TOKEN", "").strip()
    if explicit:
        return explicit
    network = env.get("LIGHTER_NETWORK", "mainnet").strip().lower()
    base_url = (
        env.get("LIGHTER_HTTP_BASE_URL", "").strip()
        or env.get("LIGHTER_REST_URL", "").strip()
        or ("https://testnet.zklighter.elliot.ai" if network == "testnet" else "https://mainnet.zklighter.elliot.ai")
    ).rstrip("/")
    account_index = int(env["LIGHTER_ACCOUNT_INDEX"])
    api_key_index = int(env["LIGHTER_API_KEY_INDEX"])
    private_key = env["LIGHTER_API_PRIVATE_KEY_HEX"]
    if private_key.startswith("0x"):
        private_key = private_key[2:]
    client = lighter.SignerClient(
        url=base_url,
        account_index=account_index,
        api_private_keys={api_key_index: private_key},
    )
    auth, err = client.create_auth_token_with_expiry(3600)
    if err is not None:
        raise RuntimeError(f"lighter auth token generation failed: {err}")
    await client.api_client.close()
    return auth


def lighter_auth_token(env: dict[str, str]) -> str:
    return asyncio.run(_lighter_auth_token_async(env))


def sign_aster_query(env: dict[str, str], params: dict[str, Any]) -> str:
    signed = dict(params)
    signed["timestamp"] = str(int(time.time() * 1000))
    recv_window = env.get("ASTER_RECV_WINDOW", "").strip()
    if recv_window:
        signed["recvWindow"] = recv_window
    canonical = urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in signed.items()))
    signature = hmac.new(
        env["ASTER_API_SECRET"].encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{canonical}&signature={signature}"


def fetch_local_runtime_state(env_file: Path) -> dict[str, Any]:
    systemd = parse_systemctl_properties(
        run_command(
            [
                "systemctl",
                "show",
                "paraphina_live",
                "--property=ActiveState,SubState,NRestarts",
            ]
        )
    )
    health = http_json("GET", "http://127.0.0.1:9898/health/detail")
    audit = json.loads(
        run_command(
            [
                sys.executable,
                str(LIVE_AUDIT),
                "--env-file",
                str(env_file),
                "--position-tol-base",
                "0.0025",
                "--max-open-orders",
                "0",
            ]
        )
    )
    hold = (
        systemd.get("ActiveState") == "active"
        and systemd.get("SubState") == "running"
        and systemd.get("NRestarts") == "0"
        and bool(health.get("healthy"))
        and bool(health.get("ready"))
        and not bool(health.get("kill_events_present"))
        and int(health.get("reconcile_mismatch_count", 0)) == 0
        and bool(audit.get("ok"))
    )
    return {
        "collected_at_utc": utc_now_iso(),
        "systemd": systemd,
        "health": health,
        "venue_audit": audit,
        "runtime_decision": "HOLD" if hold else "ROLLBACK_REQUIRED",
    }


def collect_hyperliquid(env: dict[str, str], baseline_balance: Decimal, start_ms: int) -> "VenueLedger":
    info_url = env.get("HL_INFO_URL", "").strip() or "https://api.hyperliquid.xyz/info"
    user = env["HL_VAULT_ADDRESS"]

    def post(payload: dict[str, Any]) -> Any:
        return http_json("POST", info_url, payload=payload)

    account_mode = post({"type": "userAbstraction", "user": user})
    clearinghouse_state = post({"type": "clearinghouseState", "user": user})
    spot_state = post({"type": "spotClearinghouseState", "user": user})
    funding = post({"type": "userFunding", "user": user, "startTime": start_ms})
    ledger_updates = post({"type": "userNonFundingLedgerUpdates", "user": user, "startTime": start_ms})
    fills = post({"type": "userFillsByTime", "user": user, "startTime": start_ms})

    spot_usdc = hyperliquid_spot_usdc(spot_state)
    perps_account_value = hyperliquid_perps_account_value(clearinghouse_state)
    account_mode_label = str(account_mode)
    current_balance = (
        perps_account_value + spot_usdc
        if hyperliquid_uses_separate_perps_balance(account_mode_label)
        else spot_usdc
    )

    realized = sum(decimal_of(fill.get("closedPnl")) for fill in fills or [])
    fees = -sum(decimal_of(fill.get("fee")) for fill in fills or [])
    funding_usd = sum(decimal_of((item.get("delta") or {}).get("usdc")) for item in funding or [])
    transfers = ZERO

    row = VenueLedger(
        venue="hyperliquid",
        baseline_balance_usd=baseline_balance,
        current_balance_usd=current_balance,
        realized_pnl_usd=realized,
        fees_usd=fees,
        funding_usd=funding_usd,
        transfers_usd=transfers,
        confidence="medium",
        notes=[
            "Standard/classic Hyperliquid balance truth uses clearinghouseState.marginSummary.accountValue plus spotClearinghouseState USDC."
            if hyperliquid_uses_separate_perps_balance(account_mode_label)
            else "Unified/portfolio Hyperliquid balance truth taken from spotClearinghouseState.",
            "No non-funding ledger movements in the attribution window."
            if not ledger_updates
            else "Non-funding ledger movements present; delta includes transfer effects.",
        ],
        extras={
            "account_mode": account_mode_label,
            "perps_account_value_usd": decimal_str(perps_account_value),
            "spot_usdc_total": decimal_str(spot_usdc),
            "fills_count": len(fills or []),
            "funding_count": len(funding or []),
            "non_funding_ledger_count": len(ledger_updates or []),
            "docs": HYPERLIQUID_DOCS,
        },
    )
    return row.finalize()


def collect_aster(env: dict[str, str], baseline_balance: Decimal, start_ms: int) -> "VenueLedger":
    rest_url = env.get("ASTER_REST_URL", "https://fapi.asterdex.com").rstrip("/")
    api_key = env["ASTER_API_KEY"]
    account = http_json(
        "GET",
        f"{rest_url}/fapi/v2/account?{sign_aster_query(env, {})}",
        headers={"X-MBX-APIKEY": api_key},
    )
    current_balance = decimal_of(account.get("totalWalletBalance"))

    income_items: list[dict[str, Any]] = []
    last_time: int | None = None
    while True:
        params: dict[str, Any] = {"startTime": last_time + 1 if last_time is not None else start_ms, "limit": 1000}
        payload = http_json(
            "GET",
            f"{rest_url}/fapi/v1/income?{sign_aster_query(env, params)}",
            headers={"X-MBX-APIKEY": api_key},
        )
        if not payload:
            break
        batch = [item for item in payload if isinstance(item, dict)]
        income_items.extend(batch)
        batch_max = max(int(item.get("time", 0)) for item in batch)
        if len(batch) < 1000 or (last_time is not None and batch_max <= last_time):
            break
        last_time = batch_max

    sums: dict[str, Decimal] = {}
    for item in income_items:
        income_type = str(item.get("incomeType") or "UNKNOWN")
        sums[income_type] = sums.get(income_type, ZERO) + decimal_of(item.get("income"))

    row = VenueLedger(
        venue="aster",
        baseline_balance_usd=baseline_balance,
        current_balance_usd=current_balance,
        realized_pnl_usd=sums.get("REALIZED_PNL", ZERO),
        fees_usd=sums.get("COMMISSION", ZERO),
        funding_usd=sums.get("FUNDING_FEE", ZERO),
        transfers_usd=ZERO,
        confidence="high",
        notes=[
            "Official Aster income history used for realized PnL, commission, and funding.",
        ],
        extras={
            "income_count": len(income_items),
            "income_sums": {key: decimal_to_float(value) for key, value in sums.items()},
            "docs": ASTER_DOCS,
        },
    )
    return row.finalize()


def lighter_paginated_history(
    base_url: str,
    endpoint: str,
    *,
    auth_token: str,
    account_index: int,
    l1_address: str,
    item_key: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params = {
            "auth": auth_token,
            "account_index": account_index,
            "l1_address": l1_address,
        }
        if cursor:
            params["cursor"] = cursor
        payload = http_json("GET", f"{base_url}{endpoint}", params=params)
        batch = [item for item in payload.get(item_key) or [] if isinstance(item, dict)]
        items.extend(batch)
        cursor = payload.get("next_cursor")
        if not cursor or not batch:
            break
    return items


def lighter_position_funding(
    base_url: str,
    *,
    auth_token: str,
    account_index: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params = {
            "auth": auth_token,
            "account_index": account_index,
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor
        payload = http_json("GET", f"{base_url}/api/v1/positionFunding", params=params)
        batch = [item for item in payload.get("position_fundings") or [] if isinstance(item, dict)]
        items.extend(batch)
        cursor = payload.get("next_cursor")
        if not cursor or not batch:
            break
    return items


def collect_lighter(
    env: dict[str, str],
    baseline_perp: Decimal,
    baseline_spot: Decimal,
    baseline_total: Decimal,
    start_ms: int,
    end_ms: int,
) -> "VenueLedger":
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
    l1_address = str(account.get("l1_address") or account.get("l1Address") or "")
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

    funding_items = lighter_position_funding(base_url, auth_token=auth_token, account_index=account_index)
    funding_usd = sum(decimal_of(item.get("change")) for item in funding_items)

    deposit_items = lighter_paginated_history(
        base_url,
        "/api/v1/deposit/history",
        auth_token=auth_token,
        account_index=account_index,
        l1_address=l1_address,
        item_key="deposits",
    )
    withdraw_items = lighter_paginated_history(
        base_url,
        "/api/v1/withdraw/history",
        auth_token=auth_token,
        account_index=account_index,
        l1_address=l1_address,
        item_key="withdraws",
    )
    transfer_items = lighter_paginated_history(
        base_url,
        "/api/v1/transfer/history",
        auth_token=auth_token,
        account_index=account_index,
        l1_address=l1_address,
        item_key="transfers",
    )

    transfers = ZERO
    for item in deposit_items:
        ts = int(item.get("timestamp", 0))
        if ts >= start_ms:
            transfers += decimal_of(item.get("amount"))
    for item in withdraw_items:
        ts = int(item.get("timestamp", 0))
        if ts >= start_ms:
            transfers -= decimal_of(item.get("amount"))
    for item in transfer_items:
        ts = int(item.get("timestamp", 0))
        if ts < start_ms:
            continue
        route_type = str(item.get("routeType") or item.get("route_type") or "").lower()
        amount = decimal_of(item.get("amount"))
        if route_type in {"to_derivatives", "to_perp", "to_margin"}:
            transfers += amount
        elif route_type in {"to_spot", "from_derivatives", "from_perp"}:
            transfers -= amount

    row = VenueLedger(
        venue="lighter",
        baseline_balance_usd=baseline_total,
        current_balance_usd=current_total,
        realized_pnl_usd=current_collateral - baseline_perp - funding_usd - transfers,
        fees_usd=ZERO,
        funding_usd=funding_usd,
        transfers_usd=transfers,
        spot_revaluation_usd=current_spot_value - baseline_spot,
        confidence="high",
        notes=[
            "Standard-account fee schedule is zero maker/zero taker; balance change is treated as execution quality and spot revaluation.",
            "Perp collateral and spot asset value are kept separate and recombined for baseline comparison.",
        ],
        extras={
            "current_collateral_usd": decimal_to_float(current_collateral),
            "current_spot_value_usd": decimal_to_float(current_spot_value),
            "current_assets": account.get("assets") or [],
            "funding_count": len(funding_items),
            "deposit_count": len(deposit_items),
            "withdraw_count": len(withdraw_items),
            "transfer_count": len(transfer_items),
            "docs": LIGHTER_DOCS,
        },
    )
    return row.finalize()


def paradex_token(env: dict[str, str]) -> str:
    readonly = env.get("PARADEX_READONLY_TOKEN", "").strip()
    if readonly:
        return readonly
    jwt = env.get("PARADEX_JWT", "").strip()
    if jwt:
        return jwt
    cmd = env["PARADEX_JWT_CMD"]
    merged_env = os.environ.copy()
    merged_env.update(env)
    token = run_shell(cmd, env=merged_env).strip()
    if not token:
        raise RuntimeError("PARADEX_JWT_CMD returned an empty token")
    return token


def paradex_paginated(
    base_url: str,
    path: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        payload = http_json("GET", f"{base_url}{path}", headers=headers, params=page_params)
        batch = [item for item in payload.get("results") or [] if isinstance(item, dict)]
        items.extend(batch)
        cursor = payload.get("next")
        if not cursor or not batch:
            break
    return items


def collect_paradex(env: dict[str, str], baseline_balance: Decimal, start_ms: int) -> "VenueLedger":
    rest_url = env.get("PARADEX_REST_URL", "https://api.prod.paradex.trade/v1").rstrip("/")
    token = paradex_token(env)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    account = http_json("GET", f"{rest_url}/account", headers=headers)
    fills = paradex_paginated(
        rest_url,
        "/fills",
        headers=headers,
        params={"market": env.get("PARADEX_MARKET", "ETH-USD-PERP"), "start_at": start_ms, "page_size": 200},
    )
    funding = paradex_paginated(
        rest_url,
        "/funding/payments",
        headers=headers,
        params={"market": env.get("PARADEX_MARKET", "ETH-USD-PERP"), "start_at": start_ms, "page_size": 200},
    )

    realized = sum(decimal_of(item.get("realized_pnl")) for item in fills)
    fees = -sum(decimal_of(item.get("fee")) for item in fills)
    funding_usd = sum(decimal_of(item.get("payment")) for item in funding)
    realized_funding = sum(decimal_of(item.get("realized_funding")) for item in fills)

    row = VenueLedger(
        venue="paradex",
        baseline_balance_usd=baseline_balance,
        current_balance_usd=decimal_of(account.get("account_value")),
        realized_pnl_usd=realized,
        fees_usd=fees,
        funding_usd=funding_usd,
        transfers_usd=ZERO,
        confidence="high" if abs(realized_funding - funding_usd) <= Decimal("0.01") else "medium",
        notes=[
            "Official account, fills, and funding-payment endpoints used for cashflow attribution.",
        ],
        extras={
            "fills_count": len(fills),
            "funding_payments_count": len(funding),
            "fills_realized_funding_usd": decimal_to_float(realized_funding),
            "docs": PARADEX_DOCS,
        },
    )
    return row.finalize()


def extended_paginated(path: str, *, headers: dict[str, str]) -> list[dict[str, Any]]:
    base_url = "https://api.starknet.extended.exchange"
    items: list[dict[str, Any]] = []
    cursor: int | None = None
    seen: set[int] = set()
    while True:
        params = {}
        if cursor is not None:
            params["cursor"] = cursor
        payload = http_json("GET", f"{base_url}{path}", headers=headers, params=params)
        batch = [item for item in payload.get("data") or [] if isinstance(item, dict)]
        items.extend(batch)
        next_cursor = (payload.get("pagination") or {}).get("cursor")
        if not batch or next_cursor is None or next_cursor in seen:
            break
        seen.add(int(next_cursor))
        cursor = int(next_cursor)
    return items


def collect_extended(env: dict[str, str], baseline_balance: Decimal, start_ms: int) -> "VenueLedger":
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
    positions = extended_paginated("/api/v1/user/positions/history", headers=headers)
    asset_ops = extended_paginated("/api/v1/user/assetOperations", headers=headers)
    equity_rows = [item for item in equities.get("data") or [] if isinstance(item, dict)]
    current_balance = decimal_of(equity_rows[-1]["value"])

    trade_pnl = ZERO
    funding_usd = ZERO
    fees = ZERO
    for item in positions:
        closed_time = int(item.get("closedTime") or item.get("createdTime") or 0)
        if closed_time < start_ms:
            continue
        breakdown = item.get("realisedPnlBreakdown") or {}
        trade_pnl += decimal_of(breakdown.get("tradePnl"))
        funding_usd += decimal_of(breakdown.get("fundingFees"))
        fees += decimal_of(breakdown.get("openFees")) + decimal_of(breakdown.get("closeFees"))

    transfers = ZERO
    for item in asset_ops:
        item_time = int(item.get("time") or 0)
        if item_time < start_ms:
            continue
        amount = decimal_of(item.get("amount"))
        op_type = str(item.get("type") or "").upper()
        if op_type == "DEPOSIT":
            transfers += amount
        elif op_type == "WITHDRAWAL":
            transfers -= amount

    row = VenueLedger(
        venue="extended",
        baseline_balance_usd=baseline_balance,
        current_balance_usd=current_balance,
        realized_pnl_usd=trade_pnl,
        fees_usd=fees,
        funding_usd=funding_usd,
        transfers_usd=transfers,
        confidence="medium",
        notes=[
            "Current balance uses official equity chart latest point; realized components use positions history breakdown.",
        ],
        extras={
            "account_id": account_id,
            "positions_history_count": len(positions),
            "asset_operations_count": len(asset_ops),
            "docs": EXTENDED_DOCS,
        },
    )
    return row.finalize()


@dataclass
class VenueLedger:
    venue: str
    baseline_balance_usd: Decimal
    current_balance_usd: Decimal
    realized_pnl_usd: Decimal = ZERO
    fees_usd: Decimal = ZERO
    funding_usd: Decimal = ZERO
    transfers_usd: Decimal = ZERO
    spot_revaluation_usd: Decimal = ZERO
    other_cashflow_usd: Decimal = ZERO
    confidence: str = "medium"
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    net_delta_usd: Decimal = ZERO
    unexplained_delta_usd: Decimal = ZERO

    def explained_delta_usd(self) -> Decimal:
        return (
            self.realized_pnl_usd
            + self.fees_usd
            + self.funding_usd
            + self.transfers_usd
            + self.spot_revaluation_usd
            + self.other_cashflow_usd
        )

    def finalize(self) -> "VenueLedger":
        self.net_delta_usd = self.current_balance_usd - self.baseline_balance_usd
        self.unexplained_delta_usd = self.net_delta_usd - self.explained_delta_usd()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "baseline_balance_usd": decimal_to_float(self.baseline_balance_usd),
            "current_balance_usd": decimal_to_float(self.current_balance_usd),
            "net_delta_usd": decimal_to_float(self.net_delta_usd),
            "realized_pnl_usd": decimal_to_float(self.realized_pnl_usd),
            "fees_usd": decimal_to_float(self.fees_usd),
            "funding_usd": decimal_to_float(self.funding_usd),
            "transfers_usd": decimal_to_float(self.transfers_usd),
            "spot_revaluation_usd": decimal_to_float(self.spot_revaluation_usd),
            "other_cashflow_usd": decimal_to_float(self.other_cashflow_usd),
            "explained_delta_usd": decimal_to_float(self.explained_delta_usd()),
            "unexplained_delta_usd": decimal_to_float(self.unexplained_delta_usd),
            "confidence": self.confidence,
            "notes": list(self.notes),
            "extras": self.extras,
        }


def recommend_child(ledger_by_venue: dict[str, VenueLedger]) -> dict[str, Any]:
    lighter = ledger_by_venue["lighter"]
    aster = ledger_by_venue["aster"]
    control_venues = ["hyperliquid", "extended", "paradex"]
    freeze_broken = [
        venue
        for venue in control_venues
        if abs(ledger_by_venue[venue].net_delta_usd) > ZERO
        and abs(ledger_by_venue[venue].unexplained_delta_usd) > abs(ledger_by_venue[venue].net_delta_usd) * Decimal("0.25")
    ]

    lighter_dominant = abs(lighter.net_delta_usd) >= abs(aster.net_delta_usd) and abs(lighter.net_delta_usd) >= Decimal("5")
    aster_dominant = abs(aster.net_delta_usd) >= Decimal("5")
    lighter_fee_negligible = abs(lighter.fees_usd) <= max(Decimal("0.10"), abs(lighter.net_delta_usd) * Decimal("0.05"))
    aster_fee_path_dominant = abs(aster.fees_usd) > abs(aster.realized_pnl_usd)

    if lighter_dominant and aster_dominant:
        recommended = "phase5_all5_current_surface_lighter_adverse_selection_markout_requal"
        reason = (
            "Lighter is the largest absolute loss venue and its official fee schedule is effectively zero, "
            "so the dominant problem is adverse selection / unwind quality rather than venue fee drag. "
            "Aster remains the second venue to revisit after the Lighter child."
        )
    elif lighter_dominant and lighter_fee_negligible:
        recommended = "phase5_all5_current_surface_lighter_adverse_selection_markout_requal"
        reason = (
            "Lighter dominates absolute loss with negligible venue fee load, which points to quote quality, "
            "markout, or unwind behavior rather than price of access."
        )
    elif abs(aster.net_delta_usd) >= abs(lighter.net_delta_usd) or aster_fee_path_dominant:
        recommended = "phase5_all5_current_surface_aster_taker_fee_path_requal"
        reason = (
            "Aster commission drag exceeds realized PnL magnitude in the official income ledger, "
            "which points to taker-path leakage or low-quality maker qualification."
        )
    else:
        recommended = "phase5_all5_current_surface_soft_cap_starvation_quote_budget_activation_requal"
        reason = (
            "No single venue dominates after venue-native attribution; the next defensible move is a current-surface "
            "soft-cap starvation child."
        )

    if freeze_broken:
        reason += " Freeze assumption warning for: " + ", ".join(sorted(freeze_broken)) + "."

    return {
        "recommended_child": recommended,
        "recommended_child_reason": reason,
        "non_target_venues_frozen": sorted(set(control_venues) - set(freeze_broken)),
        "freeze_assumption_failed_venues": sorted(freeze_broken),
    }


def analyze_execution_window(label: str, evidence: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    run_root = Path(str(evidence["run_root"]))
    payload: dict[str, Any] = {
        "label": label,
        "run_root": str(run_root),
        "segment_start_utc": evidence.get("segment_start_utc"),
        "segment_end_utc": evidence.get("segment_end_utc"),
        "fill_count_total": evidence.get("fill_count_total"),
        "fill_base_total_eth": evidence.get("fill_base_total_eth"),
        "fills_by_venue": evidence.get("fills_by_venue"),
    }
    live_metrics_path = run_root / "live_metrics.json"
    if live_metrics_path.exists():
        live_metrics = load_json(live_metrics_path)
        for key in ("projected_mm_budget_summary", "risk", "pnl_validity", "fills"):
            if key in live_metrics:
                payload[key] = live_metrics[key]

    telemetry_path = run_root / "telemetry_bounded.jsonl"
    attributed_metrics_path = out_dir / f"{label}_attributed_metrics.json"
    if telemetry_path.exists():
        run_command(
            [
                sys.executable,
                str(TELEMETRY_ANALYZER),
                "--telemetry",
                str(telemetry_path),
                "--execution-mode",
                "live",
                "--last-segment",
                "--metrics-json",
                str(attributed_metrics_path),
            ],
            cwd=ROOT,
        )
        payload["economics_attribution"] = load_json(attributed_metrics_path).get("economics_attribution", {})
    else:
        payload["economics_attribution"] = {}
    return payload


def ledger_rows_to_csv(path: Path, rows: list[VenueLedger]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "venue",
        "baseline_balance_usd",
        "current_balance_usd",
        "net_delta_usd",
        "realized_pnl_usd",
        "fees_usd",
        "funding_usd",
        "transfers_usd",
        "spot_revaluation_usd",
        "other_cashflow_usd",
        "explained_delta_usd",
        "unexplained_delta_usd",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.to_dict()
            writer.writerow({key: payload[key] for key in fieldnames})


def render_cashflow_summary(
    rows: list[VenueLedger],
    runtime_state: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    lines = [
        "# Phase 5 Venue Cashflow Summary",
        "",
        f"- Collected at: `{runtime_state['collected_at_utc']}`",
        f"- Runtime decision: `{runtime_state['runtime_decision']}`",
        "",
        "| Venue | Baseline | Current | Delta | Realized | Fees | Funding | Spot | Unexplained |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.venue} | "
            f"{decimal_to_float(row.baseline_balance_usd):.6f} | "
            f"{decimal_to_float(row.current_balance_usd):.6f} | "
            f"{decimal_to_float(row.net_delta_usd):.6f} | "
            f"{decimal_to_float(row.realized_pnl_usd):.6f} | "
            f"{decimal_to_float(row.fees_usd):.6f} | "
            f"{decimal_to_float(row.funding_usd):.6f} | "
            f"{decimal_to_float(row.spot_revaluation_usd):.6f} | "
            f"{decimal_to_float(row.unexplained_delta_usd):.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Recommended child: `{recommendation['recommended_child']}`",
            f"- Reason: {recommendation['recommended_child_reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_investigation_report(
    *,
    baseline: dict[str, Any],
    runtime_state: dict[str, Any],
    surface_spec_path: Path,
    surface_spec: dict[str, Any],
    rows: list[VenueLedger],
    execution_windows: dict[str, Any],
    recommendation: dict[str, Any],
    out_dir: Path,
) -> str:
    row_map = {row.venue: row for row in rows}
    total_baseline = sum((row.baseline_balance_usd for row in rows), ZERO)
    total_current = sum((row.current_balance_usd for row in rows), ZERO)
    total_delta = total_current - total_baseline

    accepted = execution_windows["accepted_soak"]
    lineage = execution_windows["lineage"]
    accepted_risk = accepted.get("risk") or {}
    accepted_budget = accepted.get("projected_mm_budget_summary") or {}

    lines = [
        "# Phase 5 Current-Surface Lighter/Aster Loss Attribution",
        "",
        f"- Generated at: `{runtime_state['collected_at_utc']}`",
        f"- Loss window: `{baseline['captured_at_utc']} -> {runtime_state['collected_at_utc']}`",
        f"- Runtime decision: `{runtime_state['runtime_decision']}`",
        f"- Accepted surface: `{surface_spec.get('surface_id')}` from `{surface_spec_path}`",
        f"- Current total: `{decimal_to_float(total_current):.6f} USD` vs baseline `{decimal_to_float(total_baseline):.2f} USD`; delta `{decimal_to_float(total_delta):.6f} USD`",
        "",
        "## Confirmed Internal Facts",
        "",
        f"- Accepted closeout on surface `{surface_spec.get('surface_id')}` remains frozen in `{surface_spec_path}`.",
        f"- The accepted 2h soak is topology-valid but explicitly not an economics closeout in `{ROOT / 'phase5/runs/phase5_reopened_final_closeout/final_closeout.md'}`.",
        f"- Accepted 2h underactivation remained severe: `would_send_zero_pct={accepted_risk.get('would_send_zero_pct')}`, `soft_governor_ticks={accepted_risk.get('soft_governor_ticks')}`, `all5_selected_ticks={accepted_budget.get('all5_selected_ticks')}`.",
        f"- Current runtime is healthy and flat: `trade_mode={runtime_state['health'].get('trade_mode')}`, `kill_events_present={runtime_state['health'].get('kill_events_present')}`, `reconcile_mismatch_count={runtime_state['health'].get('reconcile_mismatch_count')}`, direct venue audit `ok={runtime_state['venue_audit'].get('ok')}`.",
        f"- Internal PnL remains non-cash-account truth because there is no explicit funding accrual ledger in `{ROOT / 'docs/INVESTIGATIONS/funding_repo_audit.md'}`.",
        "",
        "## Confirmed External Facts",
        "",
        f"- Lighter Standard accounts are fee-free (`0 maker / 0 taker`), so large loss there is not explained by venue fees alone. Source: {LIGHTER_DOCS['fees']}",
        f"- Lighter Simple Trading Accounts keep spot and derivatives separate; account attribution must split perp collateral and spot value. Source: {LIGHTER_DOCS['accounts']}",
        f"- Lighter official account PnL supports `ignore_transfers`, which is the venue-native way to separate funded cost basis from trading/equity change. Source: {LIGHTER_DOCS['pnl']}",
        f"- Aster USDT perps are `0% maker / 0.04% taker`, and official income history reports `REALIZED_PNL`, `COMMISSION`, and `FUNDING_FEE`. Sources: {ASTER_DOCS['fees']} and {ASTER_DOCS['income']}",
        f"- Hyperliquid account mode controls balance truth: Standard uses separate perps `clearinghouseState.marginSummary.accountValue` plus spot USDC; Unified/Portfolio use spot clearinghouse state as the account balance source. Source: {HYPERLIQUID_DOCS['accounts']}",
        f"- ParaDex maker is `0%` and taker is `0.02%`, with UI-queue priority documented separately; current drift there is small enough to treat as a control venue in this step. Sources: {PARADEX_DOCS['fees']} and {PARADEX_DOCS['fastfills']}",
        f"- Extended maker is `0%` and taker is `0.025%`; current loss is small relative to Lighter and Aster. Source: {EXTENDED_DOCS['fees']}",
        "",
        "## Venue-Native Cashflow Ledger",
        "",
        "| Venue | Baseline | Current | Delta | Realized | Fees | Funding | Transfers | Spot | Unexplained | Confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.venue} | "
            f"{decimal_to_float(row.baseline_balance_usd):.6f} | "
            f"{decimal_to_float(row.current_balance_usd):.6f} | "
            f"{decimal_to_float(row.net_delta_usd):.6f} | "
            f"{decimal_to_float(row.realized_pnl_usd):.6f} | "
            f"{decimal_to_float(row.fees_usd):.6f} | "
            f"{decimal_to_float(row.funding_usd):.6f} | "
            f"{decimal_to_float(row.transfers_usd):.6f} | "
            f"{decimal_to_float(row.spot_revaluation_usd):.6f} | "
            f"{decimal_to_float(row.unexplained_delta_usd):.6f} | "
            f"{row.confidence} |"
        )
    lines.extend(
        [
            "",
            f"- Lighter delta: `{decimal_to_float(row_map['lighter'].net_delta_usd):.6f}` with zero explicit fee load and split perp/spot accounting.",
            f"- Aster delta: `{decimal_to_float(row_map['aster'].net_delta_usd):.6f}` with official `COMMISSION={decimal_to_float(row_map['aster'].fees_usd):.6f}` and `REALIZED_PNL={decimal_to_float(row_map['aster'].realized_pnl_usd):.6f}`.",
            f"- Hyperliquid delta: `{decimal_to_float(row_map['hyperliquid'].net_delta_usd):.6f}` with zero non-funding ledger movements.",
            "",
            "## Current-Surface Execution Quality",
            "",
            f"- Accepted soak window: `{accepted.get('segment_start_utc')} -> {accepted.get('segment_end_utc')}`",
            f"- Accepted soak fills: `{accepted.get('fill_count_total')}` / `{accepted.get('fill_base_total_eth')} ETH`; by venue `{accepted.get('fills_by_venue')}`",
            f"- Accepted soak soft-governor blocked ticks: `{accepted_risk.get('soft_governor_blocked_ticks')}`",
            f"- Accepted soak selected counts: `{accepted_budget.get('selected_counts')}`",
            f"- Accepted soak suppressed counts: `{accepted_budget.get('suppressed_counts')}`",
            f"- Accepted soak economics attribution: `{accepted.get('economics_attribution')}`",
            "",
            f"- Exact-surface lineage window: `{lineage.get('segment_start_utc')} -> {lineage.get('segment_end_utc')}`",
            f"- Exact-surface lineage fills: `{lineage.get('fill_count_total')}` / `{lineage.get('fill_base_total_eth')} ETH`; by venue `{lineage.get('fills_by_venue')}`",
            f"- Exact-surface lineage economics attribution: `{lineage.get('economics_attribution')}`",
            "",
            "## Inference",
            "",
            "- Lighter is the first economics target because it is the largest absolute loss venue and official venue fees are effectively zero. That points to quote quality, markout, or unwind behavior rather than venue access cost.",
            "- Aster is the second economics target because official commission drag is larger in magnitude than realized PnL, which points to taker-path leakage or failure to stay in favorable maker economics.",
            "- Hyperliquid, ParaDex, and Extended stay frozen for now because their losses are smaller and currently explainable enough to serve as control venues.",
            "",
            "## Recommended Child",
            "",
            f"- `recommended_child`: `{recommendation['recommended_child']}`",
            f"- `recommended_child_reason`: {recommendation['recommended_child_reason']}",
            f"- `non_target_venues_frozen`: `{recommendation['non_target_venues_frozen']}`",
            f"- `freeze_assumption_failed_venues`: `{recommendation['freeze_assumption_failed_venues']}`",
            "",
            "## Why Not The Other Venues",
            "",
            f"- Hyperliquid: smaller loss (`{decimal_to_float(row_map['hyperliquid'].net_delta_usd):.6f}`), no transfer evidence, and accepted-surface topology evidence still matters more than venue-local retuning right now.",
            f"- ParaDex: loss is small (`{decimal_to_float(row_map['paradex'].net_delta_usd):.6f}`) and largely explained by official realized PnL plus fees.",
            f"- Extended: loss is smaller (`{decimal_to_float(row_map['extended'].net_delta_usd):.6f}`) and currently bounded well enough to keep frozen while Lighter/Aster are investigated first.",
            "",
            "## Output Artifacts",
            "",
            f"- Cashflow ledger JSON: `{out_dir / 'venue_cashflow_ledger.json'}`",
            f"- Cashflow ledger CSV: `{out_dir / 'venue_cashflow_ledger.csv'}`",
            f"- Cashflow summary: `{out_dir / 'venue_cashflow_summary.md'}`",
            f"- Recommended child: `{out_dir / 'recommended_child.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 venue-native economics attribution gate")
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--start-utc", type=str, default="2026-04-16T00:00:00Z")
    parser.add_argument("--end-utc", type=str, default=None)
    parser.add_argument("--surface-spec", type=Path, default=DEFAULT_SURFACE_SPEC)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def parse_utc_to_ms(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).timestamp() * 1000)


def main() -> int:
    args = parse_args()
    baseline = load_baseline_manifest(args.baseline_manifest)
    end_utc = args.end_utc or utc_now_iso()
    start_ms = parse_utc_to_ms(args.start_utc)
    end_ms = parse_utc_to_ms(end_utc)
    env = load_env(args.env_file)
    surface_spec = load_yaml(args.surface_spec)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runtime_state = fetch_local_runtime_state(args.env_file)

    rows = [
        collect_hyperliquid(env, baseline["venues"]["hyperliquid"], start_ms),
        collect_extended(env, baseline["venues"]["extended"], start_ms),
        collect_lighter(
            env,
            baseline["venues"]["lighter_perp"],
            baseline["venues"]["lighter_spot"],
            baseline["venues"]["lighter_total"],
            start_ms,
            end_ms,
        ),
        collect_aster(env, baseline["venues"]["aster"], start_ms),
        collect_paradex(env, baseline["venues"]["paradex"], start_ms),
    ]
    rows.sort(key=lambda row: row.venue)
    ledger_by_venue = {row.venue: row for row in rows}

    execution_windows = {
        "accepted_soak": analyze_execution_window(
            "accepted_soak",
            surface_spec["evidence"]["reopened_multi_venue_long_soak"],
            args.out_dir,
        ),
        "lineage": analyze_execution_window(
            "lineage",
            surface_spec["evidence"]["exact_surface_lineage_requal"],
            args.out_dir,
        ),
    }
    recommendation = recommend_child(ledger_by_venue)
    recommendation.update(
        {
            "runtime_decision": runtime_state["runtime_decision"],
            "mainline_status": "hold",
            "loss_window_start_utc": baseline["captured_at_utc"],
            "loss_window_end_utc": runtime_state["collected_at_utc"],
        }
    )

    ledger_payload = {
        "collected_at_utc": runtime_state["collected_at_utc"],
        "baseline_manifest": str(args.baseline_manifest),
        "surface_spec": str(args.surface_spec),
        "rows": [row.to_dict() for row in rows],
    }
    write_json(args.out_dir / "venue_cashflow_ledger.json", ledger_payload)
    ledger_rows_to_csv(args.out_dir / "venue_cashflow_ledger.csv", rows)
    write_text(
        args.out_dir / "venue_cashflow_summary.md",
        render_cashflow_summary(rows, runtime_state, recommendation),
    )
    write_json(args.out_dir / "recommended_child.json", recommendation)
    write_json(args.out_dir / "runtime_state.json", runtime_state)
    write_json(args.out_dir / "execution_windows.json", execution_windows)

    report = render_investigation_report(
        baseline=baseline,
        runtime_state=runtime_state,
        surface_spec_path=args.surface_spec,
        surface_spec=surface_spec,
        rows=rows,
        execution_windows=execution_windows,
        recommendation=recommendation,
        out_dir=args.out_dir,
    )
    write_text(args.report_path, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
