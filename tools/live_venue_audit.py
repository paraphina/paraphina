#!/usr/bin/env python3
"""Direct venue audit for pre-canary live deployment checks.

This tool reads a promoted/current env file, queries the configured live venues
directly, and verifies that venue-side inventory and open orders are clean
enough to enter live canary.

All quantities are reported in base units. In the current all-5 deployment that
means ETH, even if some legacy env/config names still include `*_tao`.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


class AuditError(RuntimeError):
    """Raised when a venue audit cannot be completed safely."""


@dataclass
class VenueAuditResult:
    venue: str
    market: str
    position_base: float = 0.0
    open_order_count: Optional[int] = None
    open_order_count_known: bool = False
    ok: bool = False
    violations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit direct venue-side state before entering live canary."
    )
    parser.add_argument(
        "--env-file",
        required=True,
        help="Promoted/current env file to read.",
    )
    parser.add_argument(
        "--position-tol-base",
        type=float,
        default=0.0025,
        help="Maximum allowed absolute position in base units per venue.",
    )
    parser.add_argument(
        "--max-open-orders",
        type=int,
        default=0,
        help="Maximum allowed open orders per venue.",
    )
    parser.add_argument(
        "--allow-unknown-open-orders",
        action="store_true",
        help="Allow venues whose direct open-order count cannot be determined.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Per-request / per-subprocess timeout.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        env[key.strip()] = value
    return env


def merged_env(env_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(env_file))
    return env


def require_env(env: dict[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise AuditError(f"missing required env: {key}")
    return value


def env_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise AuditError(f"invalid float for {key}: {raw}") from exc


def env_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise AuditError(f"invalid integer for {key}: {raw}") from exc


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def http_json(
    method: str,
    url: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float,
) -> Any:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method.upper(), headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AuditError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise AuditError(f"{method} {url} failed: {exc.reason}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{method} {url} returned invalid JSON: {exc}") from exc


def run_json_command(cmd_text: str, *, env: dict[str, str], timeout: float) -> Any:
    cmd = shlex.split(cmd_text)
    if not cmd:
        raise AuditError("empty command")
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuditError(f"command timed out: {cmd_text}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise AuditError(
            f"command failed ({proc.returncode}): {cmd_text}"
            + (f" stderr={stderr}" if stderr else "")
        )
    stdout = proc.stdout.strip()
    if not stdout:
        raise AuditError(f"command returned empty output: {cmd_text}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"command returned invalid JSON: {cmd_text}: {exc}") from exc


def run_text_command(cmd_text: str, *, env: dict[str, str], timeout: float) -> str:
    cmd = shlex.split(cmd_text)
    if not cmd:
        raise AuditError("empty command")
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuditError(f"command timed out: {cmd_text}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise AuditError(
            f"command failed ({proc.returncode}): {cmd_text}"
            + (f" stderr={stderr}" if stderr else "")
        )
    return proc.stdout.strip()


def count_open_orders_unknown(result: VenueAuditResult, allow_unknown: bool) -> None:
    if result.open_order_count_known:
        return
    if allow_unknown:
        result.ok = len(result.violations) == 0 and len(result.errors) == 0
        return
    result.violations.append("open order count unavailable")


def finalize_result(
    result: VenueAuditResult,
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
) -> VenueAuditResult:
    if abs(result.position_base) > position_tol_base:
        result.violations.append(
            f"abs(position_base)={abs(result.position_base):.8f} > {position_tol_base:.8f}"
        )
    if result.open_order_count_known:
        if (result.open_order_count or 0) > max_open_orders:
            result.violations.append(
                f"open_order_count={result.open_order_count} > {max_open_orders}"
            )
    else:
        count_open_orders_unknown(result, allow_unknown_open_orders)
    result.ok = len(result.violations) == 0 and len(result.errors) == 0
    return result


def matching_symbol(candidate: str, target: str) -> bool:
    return candidate.strip().upper() == target.strip().upper()


def lighter_symbol_matches_market(symbol: str, market: str) -> bool:
    symbol_norm = symbol.strip().upper()
    market_norm = market.strip().upper()
    if not symbol_norm or not market_norm:
        return False
    if symbol_norm == market_norm:
        return True
    base = market_norm.split("-", 1)[0]
    return symbol_norm == base


def default_hl_info_url(env: dict[str, str]) -> str:
    network = env.get("HL_NETWORK", "mainnet").strip().lower()
    if network == "testnet":
        return "https://api.hyperliquid-testnet.xyz/info"
    return "https://api.hyperliquid.xyz/info"


def audit_hyperliquid(
    env: dict[str, str],
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    timeout: float,
) -> VenueAuditResult:
    coin = env.get("HL_COIN", "ETH").strip() or "ETH"
    result = VenueAuditResult(venue="hyperliquid", market=coin)
    try:
        user = require_env(env, "HL_VAULT_ADDRESS")
        info_url = env.get("HL_INFO_URL", "").strip() or default_hl_info_url(env)
        account = http_json(
            "POST",
            info_url,
            payload={"type": "clearinghouseState", "user": user},
            timeout=timeout,
        )
        positions = account.get("assetPositions") or []
        position_base = 0.0
        for entry in positions:
            pos = entry.get("position", entry)
            symbol = str(pos.get("coin") or pos.get("symbol") or "").strip()
            if matching_symbol(symbol, coin):
                position_base += parse_float(pos.get("szi", pos.get("size")))
        result.position_base = position_base

        order_payloads = (
            {"type": "openOrders", "user": user},
            {"type": "frontendOpenOrders", "user": user},
        )
        last_error: Optional[AuditError] = None
        for payload in order_payloads:
            try:
                orders = http_json(
                    "POST",
                    info_url,
                    payload=payload,
                    timeout=timeout,
                )
                order_list = orders if isinstance(orders, list) else orders.get("orders") or []
                result.open_order_count = sum(
                    1
                    for order in order_list
                    if matching_symbol(str(order.get("coin") or order.get("symbol") or ""), coin)
                )
                result.open_order_count_known = True
                last_error = None
                break
            except AuditError as exc:
                last_error = exc
        if last_error is not None:
            result.errors.append(str(last_error))
    except AuditError as exc:
        result.errors.append(str(exc))
    return finalize_result(
        result,
        position_tol_base=position_tol_base,
        max_open_orders=max_open_orders,
        allow_unknown_open_orders=allow_unknown_open_orders,
    )


def sign_aster_query(env: dict[str, str], params: dict[str, str]) -> str:
    api_secret = require_env(env, "ASTER_API_SECRET")
    signed = dict(params)
    signed["timestamp"] = str(int(time.time() * 1000))
    recv_window = env.get("ASTER_RECV_WINDOW", "").strip()
    if recv_window:
        signed["recvWindow"] = recv_window
    items = sorted(signed.items(), key=lambda item: item[0])
    canonical = urllib.parse.urlencode(items)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{canonical}&signature={signature}"


def audit_aster(
    env: dict[str, str],
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    timeout: float,
) -> VenueAuditResult:
    market = env.get("ASTER_MARKET", "ETHUSDT").strip() or "ETHUSDT"
    result = VenueAuditResult(venue="aster", market=market)
    try:
        rest_url = env.get("ASTER_REST_URL", "https://fapi.asterdex.com").rstrip("/")
        api_key = require_env(env, "ASTER_API_KEY")
        account_query = sign_aster_query(env, {})
        account = http_json(
            "GET",
            f"{rest_url}/fapi/v2/account?{account_query}",
            headers={"X-MBX-APIKEY": api_key},
            timeout=timeout,
        )
        result.position_base = sum(
            parse_float(pos.get("positionAmt"))
            for pos in account.get("positions") or []
            if matching_symbol(str(pos.get("symbol") or ""), market)
        )
        orders_query = sign_aster_query(env, {"symbol": market})
        orders = http_json(
            "GET",
            f"{rest_url}/fapi/v1/openOrders?{orders_query}",
            headers={"X-MBX-APIKEY": api_key},
            timeout=timeout,
        )
        result.open_order_count = sum(
            1
            for order in (orders or [])
            if matching_symbol(str(order.get("symbol") or ""), market)
        )
        result.open_order_count_known = True
    except AuditError as exc:
        result.errors.append(str(exc))
    return finalize_result(
        result,
        position_tol_base=position_tol_base,
        max_open_orders=max_open_orders,
        allow_unknown_open_orders=allow_unknown_open_orders,
    )


def audit_lighter(
    env: dict[str, str],
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    timeout: float,
) -> VenueAuditResult:
    market = env.get("LIGHTER_MARKET", "ETH-USD").strip() or "ETH-USD"
    result = VenueAuditResult(venue="lighter", market=market)
    try:
        network = env.get("LIGHTER_NETWORK", "mainnet").strip().lower()
        default_rest = (
            "https://testnet.zklighter.elliot.ai"
            if network == "testnet"
            else "https://mainnet.zklighter.elliot.ai"
        )
        rest_url = (
            env.get("LIGHTER_HTTP_BASE_URL", "").strip()
            or env.get("LIGHTER_REST_URL", "").strip()
            or default_rest
        ).rstrip("/")
        account_index = require_env(env, "LIGHTER_ACCOUNT_INDEX")
        query = urllib.parse.urlencode({"by": "index", "value": account_index})
        headers: dict[str, str] = {}
        auth_token = env.get("LIGHTER_AUTH_TOKEN", "").strip()
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        account = http_json(
            "GET",
            f"{rest_url}/api/v1/account?{query}",
            headers=headers,
            timeout=timeout,
        )
        nested_account = None
        if isinstance(account, dict):
            accounts = account.get("accounts")
            if isinstance(accounts, list) and accounts:
                first = accounts[0]
                if isinstance(first, dict):
                    nested_account = first
        positions = []
        if isinstance(account, dict):
            positions = account.get("positions") or []
        if not positions and nested_account is not None:
            positions = nested_account.get("positions") or []
        position_base = 0.0
        for pos in positions:
            symbol = str(pos.get("symbol") or "").strip()
            if not lighter_symbol_matches_market(symbol, market):
                continue
            if pos.get("size") is not None:
                position_base += parse_float(pos.get("size"))
                continue
            base = parse_float(pos.get("position"))
            sign = parse_float(pos.get("sign"), 1.0)
            position_base += base * (1.0 if sign >= 0 else -1.0)
        result.position_base = position_base

        count_candidates: list[int] = []
        count_keys = (
            "open_order_count",
            "open_orders_count",
            "openOrderCount",
            "openOrdersCount",
            "pending_order_count",
            "total_order_count",
            "total_isolated_order_count",
        )
        if isinstance(account, dict):
            for key in count_keys:
                if key in account:
                    count_candidates.append(int(parse_float(account.get(key))))
        if nested_account is not None:
            for key in count_keys:
                if key in nested_account:
                    count_candidates.append(int(parse_float(nested_account.get(key))))
        if count_candidates:
            result.open_order_count = max(count_candidates)
            result.open_order_count_known = True
        else:
            for key in ("open_orders", "openOrders", "orders"):
                orders = account.get(key) if isinstance(account, dict) else None
                if not isinstance(orders, list) and nested_account is not None:
                    orders = nested_account.get(key)
                if isinstance(orders, list):
                    filtered = [
                        order for order in orders
                        if lighter_symbol_matches_market(
                            str(order.get("symbol") or order.get("market") or ""),
                            market,
                        )
                    ]
                    result.open_order_count = len(filtered)
                    result.open_order_count_known = True
                    break
    except AuditError as exc:
        result.errors.append(str(exc))
    return finalize_result(
        result,
        position_tol_base=position_tol_base,
        max_open_orders=max_open_orders,
        allow_unknown_open_orders=allow_unknown_open_orders,
    )


def paradex_token(env: dict[str, str], *, timeout: float) -> str:
    jwt = env.get("PARADEX_JWT", "").strip()
    if jwt:
        return jwt
    jwt_cmd = require_env(env, "PARADEX_JWT_CMD")
    token = run_text_command(jwt_cmd, env=env, timeout=timeout).strip()
    if not token:
        raise AuditError("PARADEX_JWT_CMD returned empty token")
    return token


def paradex_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        inner = value.get("results")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


def audit_paradex(
    env: dict[str, str],
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    timeout: float,
) -> VenueAuditResult:
    market = env.get("PARADEX_MARKET", "ETH-USD-PERP").strip() or "ETH-USD-PERP"
    result = VenueAuditResult(venue="paradex", market=market)
    try:
        rest_url = env.get("PARADEX_REST_URL", "https://api.prod.paradex.trade/v1").rstrip("/")
        account_path = env.get("PARADEX_ACCOUNT_PATH", "/account").strip() or "/account"
        order_path = env.get("PARADEX_ORDER_PATH", "/orders").strip() or "/orders"
        token = paradex_token(env, timeout=timeout)
        headers = {"Authorization": f"Bearer {token}"}

        account = http_json(
            "GET",
            f"{rest_url}{account_path}",
            headers=headers,
            timeout=timeout,
        )
        positions = paradex_list(account.get("positions") if isinstance(account, dict) else None)
        if not positions:
            positions = paradex_list(
                http_json(
                    "GET",
                    f"{rest_url}/positions?{urllib.parse.urlencode({'market': market})}",
                    headers=headers,
                    timeout=timeout,
                )
            )
        position_base = 0.0
        for pos in positions:
            symbol = str(pos.get("market") or pos.get("symbol") or "").strip()
            if not matching_symbol(symbol, market):
                continue
            size = parse_float(pos.get("size"))
            side = str(pos.get("side") or "").strip().lower()
            if side == "short" and size > 0:
                size = -size
            elif side == "long" and size < 0:
                size = abs(size)
            position_base += size
        result.position_base = position_base

        orders = paradex_list(
            http_json(
                "GET",
                f"{rest_url}{order_path}",
                headers=headers,
                timeout=timeout,
            )
        )
        open_statuses = {"OPEN", "NEW", "PARTIALLY_FILLED"}
        result.open_order_count = sum(
            1
            for order in orders
            if matching_symbol(str(order.get("market") or order.get("symbol") or ""), market)
            and str(order.get("status") or "").upper() in open_statuses
        )
        result.open_order_count_known = True
    except AuditError as exc:
        result.errors.append(str(exc))
    return finalize_result(
        result,
        position_tol_base=position_tol_base,
        max_open_orders=max_open_orders,
        allow_unknown_open_orders=allow_unknown_open_orders,
    )


def audit_extended(
    env: dict[str, str],
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    timeout: float,
) -> VenueAuditResult:
    market = env.get("EXTENDED_MARKET", "ETH-USD").strip() or "ETH-USD"
    result = VenueAuditResult(venue="extended", market=market)
    try:
        trader_cmd = require_env(env, "EXTENDED_TRADER_CMD")
        snapshot = run_json_command(
            f"{trader_cmd} snapshot --market {shlex.quote(market)}",
            env=env,
            timeout=timeout,
        )
        positions = snapshot.get("positions") if isinstance(snapshot, dict) else []
        result.position_base = sum(
            parse_float(pos.get("size"))
            for pos in (positions or [])
            if matching_symbol(str(pos.get("market") or ""), market)
        )
        orders = run_json_command(
            f"{trader_cmd} open_orders --market {shlex.quote(market)}",
            env=env,
            timeout=timeout,
        )
        result.open_order_count = len(
            [
                order
                for order in (orders or [])
                if matching_symbol(str(order.get("market") or market), market)
            ]
        )
        result.open_order_count_known = True
    except AuditError as exc:
        result.errors.append(str(exc))
    return finalize_result(
        result,
        position_tol_base=position_tol_base,
        max_open_orders=max_open_orders,
        allow_unknown_open_orders=allow_unknown_open_orders,
    )


AUDITORS = {
    "hyperliquid": audit_hyperliquid,
    "lighter": audit_lighter,
    "extended": audit_extended,
    "aster": audit_aster,
    "paradex": audit_paradex,
}


def main() -> int:
    args = parse_args()
    env_file = Path(args.env_file)
    if not env_file.exists():
        raise SystemExit(f"env file not found: {env_file}")
    env = merged_env(env_file)
    connectors_raw = env.get(
        "PARAPHINA_LIVE_CONNECTORS",
        "hyperliquid,lighter,extended,aster,paradex",
    )
    connectors = [item.strip().lower() for item in connectors_raw.split(",") if item.strip()]
    results: list[VenueAuditResult] = []
    for venue in connectors:
        auditor = AUDITORS.get(venue)
        if auditor is None:
            results.append(
                VenueAuditResult(
                    venue=venue,
                    market="",
                    ok=False,
                    violations=["unsupported venue audit"],
                )
            )
            continue
        results.append(
            auditor(
                env,
                position_tol_base=args.position_tol_base,
                max_open_orders=args.max_open_orders,
                allow_unknown_open_orders=args.allow_unknown_open_orders,
                timeout=args.timeout_seconds,
            )
        )

    violations = [
        f"{result.venue}: {message}"
        for result in results
        for message in result.violations + result.errors
    ]
    payload = {
        "ok": len(violations) == 0,
        "env_file": str(env_file),
        "connectors": connectors,
        "position_tol_base": args.position_tol_base,
        "max_open_orders": args.max_open_orders,
        "allow_unknown_open_orders": args.allow_unknown_open_orders,
        "results": [asdict(result) for result in results],
        "violations": violations,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
