#!/usr/bin/env python3
"""Phase 5.1b read-only Lighter account/native-limit evidence collector.

This tool emits schema_version=2 non-live evidence records only. It never
submits orders, never calls sendTx/sendTxBatch, and never authorizes live,
canary, capital, or risk-limit changes.

Inputs can be captured JSON files or authenticated read-only HTTP GETs. The
HTTP mode is intentionally explicit and only calls account/order/limit/trade
read endpoints needed for Phase 5.1b evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import stat
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51b_lighter_account_native_limits"
DEFAULT_BASE_URL = "https://mainnet.zklighter.elliot.ai"
DEFAULT_TESTNET_BASE_URL = "https://testnet.zklighter.elliot.ai"
READONLY_ENDPOINTS = {
    "account": "/api/v1/account",
    "account_limits": "/api/v1/accountLimits",
    "active_orders": "/api/v1/accountActiveOrders",
    "order_books": "/api/v1/orderBooks",
    "trades": "/api/v1/trades",
}
SENSITIVE_KEYS = {
    "auth",
    "authtoken",
    "authorization",
    "accesstoken",
    "api_key",
    "apikey",
    "bearer",
    "credential",
    "credentials",
    "jwt",
    "password",
    "private_key",
    "privatekey",
    "secret",
    "session",
    "signature",
    "token",
}


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, (dict, list)):
        raise ValueError(f"expected object or array JSON in {path}")
    return data


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key or not key.replace("_", "").isalnum():
                continue
            try:
                parsed = shlex.split(raw_value, comments=False, posix=True)
            except ValueError:
                parsed = [raw_value.strip().strip("'\"")]
            values[key] = parsed[0] if parsed else ""
    return values


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
            if (
                normalized in SENSITIVE_KEYS
                or "token" in normalized
                or "privatekey" in normalized
                or "secret" in normalized
                or "password" in normalized
                or "credential" in normalized
                or "authorization" in normalized
                or "signature" in normalized
                or normalized == "jwt"
            ):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _write_evidence_index(out_dir: Path, artifact_paths: list[Path], metadata: dict[str, Any]) -> Path:
    index_path = out_dir / "evidence_pack" / "artifact_index.json"
    index = {
        "schema_version": 1,
        "metadata": metadata,
        "artifacts": _artifact_infos(out_dir, artifact_paths),
    }
    _write_json(index_path, index)
    return index_path


def _write_manifest(
    out_dir: Path,
    artifact_paths: list[Path],
    metadata: dict[str, Any],
    created_utc: str,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": metadata,
        "files": _artifact_infos(out_dir, artifact_paths),
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("spec baseline_commit does not match clean Phase 5 baseline")
    if spec.get("run_mode") != "READ_ONLY":
        raise ValueError("Phase 5.1b Lighter account/native-limit run_mode must be READ_ONLY")
    if spec.get("venue_id") != "lighter":
        raise ValueError("Phase 5.1b account/native-limit capture must be Lighter-only")
    if spec.get("no_live_flag") is not True:
        raise ValueError("no_live_flag must be true")
    if spec.get("capital_escalation_flag") is not False:
        raise ValueError("capital escalation must be false")
    if spec.get("risk_limit_override_flag") is not False:
        raise ValueError("risk limit override must be false")
    constraints = spec.get("constraints", {})
    if constraints.get("live_orders_allowed") is not False:
        raise ValueError("live_orders_allowed must be false")
    if constraints.get("capital_change_allowed") is not False:
        raise ValueError("capital_change_allowed must be false")
    if constraints.get("risk_limit_relaxation_allowed") is not False:
        raise ValueError("risk_limit_relaxation_allowed must be false")
    if constraints.get("sendtx_allowed") is not False:
        raise ValueError("sendtx_allowed must be false")


def _base_event(
    *,
    event_type: str,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "event_type": event_type,
        "event_seq": event_seq,
        "timestamp_local_ns": timestamp_ns + event_seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }


def _first_value(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys:
                return value
        for value in obj.values():
            found = _first_value(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _first_value(item, keys)
            if found is not None:
                return found
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_account_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if "PREMIUM" in text:
        return "PREMIUM"
    if "STANDARD" in text or "RETAIL" in text:
        return "STANDARD"
    return "UNKNOWN"


def _extract_items(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key, value in payload.items():
        if str(key).lower() in keys:
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
    return []


def _extract_account(payload: Any) -> dict[str, Any]:
    accounts = _extract_items(payload, {"accounts", "account"})
    if accounts:
        return accounts[0]
    return payload if isinstance(payload, dict) else {}


def _extract_active_orders(payload: Any) -> list[dict[str, Any]]:
    return _extract_items(
        payload,
        {
            "orders",
            "active_orders",
            "activeorders",
            "account_active_orders",
            "accountactiveorders",
        },
    )


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(
            item.get("status")
            or item.get("order_status")
            or item.get("orderStatus")
            or "UNKNOWN"
        ).upper()
        counts[status] = counts.get(status, 0) + 1
    return counts


def _extract_market_metadata(payload: Any, market_symbol: str | None, market_id: int | None) -> dict[str, Any]:
    books = _extract_items(payload, {"order_books", "orderbooks", "markets"})
    if not books and isinstance(payload, dict):
        books = [payload]
    target_symbol = (market_symbol or "").upper().replace("-PERP", "").replace("-USD", "")
    for book in books:
        symbol = str(book.get("symbol") or book.get("market") or "").upper()
        book_market_id = _as_int(book.get("market_id") or book.get("marketId") or book.get("id"))
        normalized_symbol = symbol.replace("-PERP", "").replace("-USD", "")
        if market_id is not None and book_market_id == market_id:
            return book
        if target_symbol and normalized_symbol == target_symbol:
            return book
    return books[0] if books else {}


def _source_record(
    *,
    name: str,
    payload: dict[str, Any] | list[Any] | None,
    source_path: Path | None,
    source_endpoint: str | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "name": name,
        "payload": payload,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "source_path": str(source_path) if source_path else None,
        "source_endpoint": source_endpoint,
    }


def _load_source_file(name: str, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _source_record(
        name=name,
        payload=_load_json(path),
        source_path=path,
        source_endpoint=None,
    )


def _resolve_base_url(env: dict[str, str], explicit: str | None = None) -> str:
    if explicit:
        return explicit.rstrip("/")
    configured = env.get("LIGHTER_HTTP_BASE_URL", "").strip() or env.get("LIGHTER_REST_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    network = env.get("LIGHTER_NETWORK", "mainnet").strip().lower()
    return DEFAULT_TESTNET_BASE_URL if network == "testnet" else DEFAULT_BASE_URL


def _check_safe_sdk_entry(label: str, path: Path, *, require_dir: bool | None = None) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    if require_dir is True and not path.is_dir():
        raise RuntimeError(f"{label} is not a directory: {path}")
    if require_dir is False and not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")
    info = path.stat()
    if info.st_uid not in (0, os.getuid()):
        raise RuntimeError(f"{label} must be owned by root or current user: {path}")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(f"{label} must not be group/world writable: {path}")


def _resolve_safe_lighter_sdk_path(sdk_path: Path) -> Path:
    expanded = sdk_path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError(f"lighter-sdk path must not be a symlink: {expanded}")
    sdk_root = expanded.resolve(strict=True)
    package_dir = sdk_root / "lighter"
    for label, path in (("lighter-sdk path", sdk_root), ("lighter package path", package_dir)):
        if path.is_symlink():
            raise RuntimeError(f"{label} must not be a symlink: {path}")
        _check_safe_sdk_entry(label, path, require_dir=True)
    for current_root, dirnames, filenames in os.walk(package_dir):
        current = Path(current_root)
        _check_safe_sdk_entry("lighter package directory", current, require_dir=True)
        for dirname in dirnames:
            _check_safe_sdk_entry("lighter package directory", current / dirname, require_dir=True)
        for filename in filenames:
            _check_safe_sdk_entry("lighter package file", current / filename, require_dir=False)
    return sdk_root


def _import_lighter_sdk(sdk_path: Path | None) -> Any:
    try:
        import lighter  # type: ignore

        return lighter
    except ImportError:
        if sdk_path is not None:
            sdk_root = _resolve_safe_lighter_sdk_path(sdk_path)
            sys.path.insert(0, str(sdk_root))
            import lighter  # type: ignore

            return lighter
    raise RuntimeError("lighter-sdk is required for --allow-sdk-auth")


def _get_auth_token(env: dict[str, str], allow_sdk_auth: bool, sdk_path: Path | None) -> str:
    explicit = env.get("LIGHTER_AUTH_TOKEN", "").strip()
    if explicit:
        return explicit
    if not allow_sdk_auth:
        raise RuntimeError(
            "missing LIGHTER_AUTH_TOKEN; set it or pass --allow-sdk-auth to derive "
            "a short-lived read-only auth token from existing Lighter API key env"
        )
    lighter = _import_lighter_sdk(sdk_path)
    base_url = _resolve_base_url(env)
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
    auth, err = client.create_auth_token_with_expiry(600)
    if err is not None:
        raise RuntimeError(f"lighter auth token generation failed: {err}")
    api_client = getattr(client, "api_client", None)
    if api_client is not None and hasattr(api_client, "close"):
        asyncio.run(api_client.close())
    return auth


def _http_get_json(
    base_url: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    auth_token: str | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any] | list[Any]:
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
    url = f"{base_url.rstrip('/')}{endpoint}"
    if query:
        url = f"{url}?{query}"
    headers = {"Accept": "application/json", "User-Agent": "paraphina-phase51b-readonly/1"}
    if auth_token:
        headers["Authorization"] = auth_token
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"unexpected JSON payload from {endpoint}")
    return payload


def _fetch_readonly_sources(
    *,
    env: dict[str, str],
    base_url: str,
    account_index: int,
    market_id: int | None,
    include_trades: bool,
    allow_sdk_auth: bool,
    sdk_path: Path | None,
    timeout_s: float,
) -> dict[str, dict[str, Any]]:
    auth_token = _get_auth_token(env, allow_sdk_auth=allow_sdk_auth, sdk_path=sdk_path)
    fetched: dict[str, dict[str, Any]] = {}
    endpoint = READONLY_ENDPOINTS["account"]
    fetched["account"] = _source_record(
        name="account",
        payload=_http_get_json(base_url, endpoint, params={"by": "index", "value": account_index}, timeout_s=timeout_s),
        source_path=None,
        source_endpoint=f"{base_url}{endpoint}",
    )
    endpoint = READONLY_ENDPOINTS["account_limits"]
    fetched["account_limits"] = _source_record(
        name="account_limits",
        payload=_http_get_json(base_url, endpoint, params={"account_index": account_index}, auth_token=auth_token, timeout_s=timeout_s),
        source_path=None,
        source_endpoint=f"{base_url}{endpoint}",
    )
    endpoint = READONLY_ENDPOINTS["active_orders"]
    fetched["active_orders"] = _source_record(
        name="active_orders",
        payload=_http_get_json(
            base_url,
            endpoint,
            params={"account_index": account_index, "market_id": market_id},
            auth_token=auth_token,
            timeout_s=timeout_s,
        ),
        source_path=None,
        source_endpoint=f"{base_url}{endpoint}",
    )
    endpoint = READONLY_ENDPOINTS["order_books"]
    fetched["order_books"] = _source_record(
        name="order_books",
        payload=_http_get_json(base_url, endpoint, timeout_s=timeout_s),
        source_path=None,
        source_endpoint=f"{base_url}{endpoint}",
    )
    if include_trades:
        endpoint = READONLY_ENDPOINTS["trades"]
        fetched["trades"] = _source_record(
            name="trades",
            payload=_http_get_json(
                base_url,
                endpoint,
                params={
                    "account_index": account_index,
                    "market_id": market_id,
                    "market_type": "perp",
                    "sort_by": "timestamp",
                    "limit": 100,
                },
                auth_token=auth_token,
                timeout_s=timeout_s,
            ),
            source_path=None,
            source_endpoint=f"{base_url}{endpoint}",
        )
    return {k: v for k, v in fetched.items() if v is not None}


def _account_profile_event(
    *,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
    account_source: dict[str, Any] | None,
    order_books_source: dict[str, Any] | None,
    account_index: int | None,
    market_symbol: str | None,
    market_id: int | None,
) -> dict[str, Any]:
    account = _extract_account(account_source["payload"] if account_source else {})
    l1_address = _as_text(account.get("l1_address") or account.get("l1Address") or account.get("address"))
    observed_account_index = _as_int(account.get("account_index") or account.get("accountIndex") or account.get("index")) or account_index
    account_type = _normalize_account_type(_first_value(account, {"account_type", "accounttype", "account_tier", "accounttier", "tier"}))
    market = _extract_market_metadata(order_books_source["payload"] if order_books_source else {}, market_symbol, market_id)
    event = _base_event(event_type="V2_LIGHTER_ACCOUNT_PROFILE", event_seq=event_seq, run_id=run_id, timestamp_ns=timestamp_ns)
    event.update({
        "venue_id": "lighter",
        "account_index": observed_account_index,
        "account_id_nonsecret_hash": _stable_hash({"account_index": observed_account_index, "l1_address": l1_address}) if observed_account_index is not None or l1_address else None,
        "account_l1_address_present": bool(l1_address),
        "lighter_account_type": account_type,
        "lighter_account_profile_status": "OBSERVED" if account_type != "UNKNOWN" else "UNKNOWN",
        "market_id": _as_int(market.get("market_id") or market.get("marketId") or market.get("id")) or market_id,
        "market_symbol": _as_text(market.get("symbol") or market.get("market") or market_symbol),
        "market_metadata_status": "OBSERVED" if market else "MISSING",
        "maker_fee_raw": _as_text(market.get("maker_fee") or market.get("makerFee") or market.get("maker_fee_percent") or market.get("makerFeePercent")),
        "taker_fee_raw": _as_text(market.get("taker_fee") or market.get("takerFee") or market.get("taker_fee_percent") or market.get("takerFeePercent")),
        "maker_fee_bps": _as_float(
            market.get("maker_fee_bps")
            if market.get("maker_fee_bps") is not None
            else market.get("makerFeeBps")
        ),
        "taker_fee_bps": _as_float(
            market.get("taker_fee_bps")
            if market.get("taker_fee_bps") is not None
            else market.get("takerFeeBps")
        ),
        "price_decimals": _as_int(market.get("price_decimals") or market.get("priceDecimals")),
        "size_decimals": _as_int(market.get("size_decimals") or market.get("sizeDecimals")),
        "account_source_sha256": account_source["sha256"] if account_source else None,
        "order_books_source_sha256": order_books_source["sha256"] if order_books_source else None,
        "decision": "HOLD",
        "decision_reason_primary": "phase51b_readonly_account_profile_capture",
        "decision_reason_secondary_list": ["nonlive_readonly_evidence_only", "not_live_authorization", "not_financial_claim"],
        "admissible_for_financial_claim": False,
    })
    return event


def _account_limits_event(
    *,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
    limits_source: dict[str, Any],
    account_index: int | None,
) -> dict[str, Any]:
    payload = limits_source["payload"]
    raw_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
    event = _base_event(event_type="V2_LIGHTER_ACCOUNT_LIMITS", event_seq=event_seq, run_id=run_id, timestamp_ns=timestamp_ns)
    event.update({
        "venue_id": "lighter",
        "account_index": account_index,
        "account_limits_status": "OBSERVED",
        "account_limits_source_sha256": limits_source["sha256"],
        "account_limits_source_endpoint": limits_source["source_endpoint"],
        "account_limits_raw_keys": raw_keys,
        "sendtx_per_minute_limit": _as_int(_first_value(payload, {"sendtx_per_minute_limit", "sendtxperminutelimit", "sendtx_per_minute", "sendtxperminute", "send_tx_limit", "sendtxlimit"})),
        "sendtx_per_minute_remaining": _as_int(_first_value(payload, {"sendtx_per_minute_remaining", "sendtxperminuteremaining", "send_tx_remaining", "sendtxremaining"})),
        "rest_requests_per_minute_limit": _as_int(_first_value(payload, {"rest_requests_per_minute_limit", "restrequestsperminutelimit", "standard_requests_per_minute", "standardrequestsperminute"})),
        "weighted_requests_per_minute_limit": _as_int(_first_value(payload, {"weighted_requests_per_minute_limit", "weightedrequestsperminutelimit", "premium_weighted_requests", "premiumweightedrequests"})),
        "pending_orders_per_account_limit": _as_int(_first_value(payload, {"pending_orders_per_account_limit", "pendingordersperaccountlimit", "pending_orders_per_account", "pendingordersperaccount"})),
        "pending_orders_per_market_limit": _as_int(_first_value(payload, {"pending_orders_per_market_limit", "pendingorderspermarketlimit", "pending_orders_per_market", "pendingorderspermarket"})),
        "active_orders_per_account_limit": _as_int(_first_value(payload, {"active_orders_per_account_limit", "activeordersperaccountlimit", "active_orders_per_account", "activeordersperaccount"})),
        "active_orders_per_market_limit": _as_int(_first_value(payload, {"active_orders_per_market_limit", "activeorderspermarketlimit", "active_orders_per_market", "activeorderspermarket"})),
        "volume_quota_remaining": _as_float(_first_value(payload, {"volume_quota_remaining", "volumequotaremaining", "quota_remaining", "quotaremaining"})),
        "rate_limit_headroom_status": "OBSERVED",
        "decision": "HOLD",
        "decision_reason_primary": "phase51b_readonly_account_limits_capture",
        "decision_reason_secondary_list": ["nonlive_readonly_evidence_only", "native_limits_not_ev_admission", "not_live_authorization"],
        "admissible_for_financial_claim": False,
    })
    return event


def _active_orders_event(
    *,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
    active_orders_source: dict[str, Any],
    limits_source: dict[str, Any] | None,
    account_index: int | None,
    market_id: int | None,
) -> dict[str, Any]:
    orders = _extract_active_orders(active_orders_source["payload"])
    status_counts = _status_counts(orders)
    pending_count = sum(count for status, count in status_counts.items() if "PENDING" in status)
    market_count = 0
    if market_id is not None:
        for order in orders:
            if _as_int(order.get("market_id") or order.get("marketId")) == market_id:
                market_count += 1
    else:
        market_count = len(orders)
    limits_payload = limits_source["payload"] if limits_source else {}
    per_account_limit = _as_int(_first_value(limits_payload, {"active_orders_per_account_limit", "activeordersperaccountlimit", "active_orders_per_account", "activeordersperaccount"}))
    per_market_limit = _as_int(_first_value(limits_payload, {"active_orders_per_market_limit", "activeorderspermarketlimit", "active_orders_per_market", "activeorderspermarket"}))
    event = _base_event(event_type="V2_LIGHTER_ACTIVE_ORDERS", event_seq=event_seq, run_id=run_id, timestamp_ns=timestamp_ns)
    event.update({
        "venue_id": "lighter",
        "account_index": account_index,
        "market_id": market_id,
        "active_orders_status": "OBSERVED",
        "active_orders_source_sha256": active_orders_source["sha256"],
        "active_orders_source_endpoint": active_orders_source["source_endpoint"],
        "active_orders_count_total": len(orders),
        "active_orders_count_market": market_count,
        "pending_orders_count_total": pending_count,
        "active_order_status_keys": sorted(status_counts.keys()),
        "active_order_sample_hash": _stable_hash(orders[:20]),
        "active_orders_per_account_limit": per_account_limit,
        "active_orders_per_market_limit": per_market_limit,
        "active_order_headroom_account": per_account_limit - len(orders) if per_account_limit is not None else None,
        "active_order_headroom_market": per_market_limit - market_count if per_market_limit is not None else None,
        "open_order_limit_status": "OBSERVED" if per_account_limit is not None or per_market_limit is not None else "UNKNOWN",
        "decision": "HOLD",
        "decision_reason_primary": "phase51b_readonly_active_orders_capture",
        "decision_reason_secondary_list": ["nonlive_readonly_evidence_only", "active_orders_not_execution_authority", "not_live_authorization"],
        "admissible_for_financial_claim": False,
    })
    return event


def _trades_event(
    *,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
    trades_source: dict[str, Any],
    account_index: int | None,
    market_id: int | None,
) -> dict[str, Any]:
    trades = _extract_items(trades_source["payload"], {"trades", "trade_history", "tradehistory"})
    maker_count = 0
    taker_count = 0
    unknown_count = 0
    for trade in trades:
        role = str(trade.get("role") or trade.get("maker_taker") or trade.get("makerTaker") or trade.get("liquidity") or "").lower()
        if role == "maker" or role.startswith("maker"):
            maker_count += 1
        elif role == "taker" or role.startswith("taker"):
            taker_count += 1
        else:
            unknown_count += 1
    event = _base_event(event_type="V2_LIGHTER_TRADE_ATTRIBUTION_SAMPLE", event_seq=event_seq, run_id=run_id, timestamp_ns=timestamp_ns)
    event.update({
        "venue_id": "lighter",
        "account_index": account_index,
        "market_id": market_id,
        "trade_sample_status": "OBSERVED",
        "trade_sample_source_sha256": trades_source["sha256"],
        "trade_sample_source_endpoint": trades_source["source_endpoint"],
        "trade_sample_count": len(trades),
        "maker_trade_count": maker_count,
        "taker_trade_count": taker_count,
        "unknown_role_trade_count": unknown_count,
        "maker_taker_attribution_status": "OBSERVED" if maker_count + taker_count > 0 else "UNKNOWN",
        "decision": "HOLD",
        "decision_reason_primary": "phase51b_readonly_trade_attribution_sample",
        "decision_reason_secondary_list": ["nonlive_readonly_evidence_only", "trade_sample_not_balance_authority", "not_live_authorization"],
        "admissible_for_financial_claim": False,
    })
    return event


def _copy_sanitized_sources(out_dir: Path, sources: dict[str, dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    source_dir = out_dir / "source_snapshots"
    for name, source in sorted(sources.items()):
        path = source_dir / f"{name}.sanitized.json"
        _write_json(path, _redact(source["payload"]))
        paths.append(path)
    return paths


def run(
    *,
    spec_path: Path,
    output_root: Path | None,
    run_id: str | None,
    account_json: Path | None,
    account_limits_json: Path | None,
    active_orders_json: Path | None,
    order_books_json: Path | None,
    trades_json: Path | None,
    env_file: Path | None,
    fetch_readonly: bool,
    include_trades: bool,
    allow_sdk_auth: bool,
    lighter_sdk_path: Path | None,
    base_url: str | None,
    account_index: int | None,
    market_id: int | None,
    market_symbol: str | None,
    timestamp_ns: int | None,
    timeout_s: float,
) -> Path:
    spec = _load_json(spec_path)
    if not isinstance(spec, dict):
        raise ValueError(f"expected object JSON in {spec_path}")
    _validate_spec(spec)
    env = dict(os.environ)
    if env_file is not None:
        env.update(_load_env_file(env_file))
    run_id = run_id or f"{spec.get('experiment_id', 'PHASE51B-LIGHTER-NATIVE-LIMITS')}_{_utc_stamp()}"
    output_root = output_root or Path(spec.get("output_root", DEFAULT_OUTPUT_ROOT))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)
    resolved_account_index = account_index
    if resolved_account_index is None:
        configured = spec.get("account_index") or env.get("LIGHTER_ACCOUNT_INDEX")
        resolved_account_index = int(configured) if configured not in (None, "") else None
    resolved_market_id = market_id
    if resolved_market_id is None:
        configured_market_id = spec.get("market_id") or env.get("LIGHTER_MARKET_ID")
        resolved_market_id = int(configured_market_id) if configured_market_id not in (None, "") else None
    resolved_market_symbol = market_symbol or spec.get("market_symbol") or env.get("LIGHTER_MARKET")
    resolved_base_url = _resolve_base_url(env, base_url or spec.get("base_url"))

    sources: dict[str, dict[str, Any]] = {
        key: value
        for key, value in {
            "account": _load_source_file("account", account_json),
            "account_limits": _load_source_file("account_limits", account_limits_json),
            "active_orders": _load_source_file("active_orders", active_orders_json),
            "order_books": _load_source_file("order_books", order_books_json),
            "trades": _load_source_file("trades", trades_json),
        }.items()
        if value is not None
    }
    if fetch_readonly:
        if resolved_account_index is None:
            raise ValueError("fetch-readonly requires LIGHTER_ACCOUNT_INDEX or --account-index")
        sources.update(_fetch_readonly_sources(
            env=env,
            base_url=resolved_base_url,
            account_index=resolved_account_index,
            market_id=resolved_market_id,
            include_trades=include_trades,
            allow_sdk_auth=allow_sdk_auth,
            sdk_path=lighter_sdk_path,
            timeout_s=timeout_s,
        ))
    if not sources:
        raise ValueError("provide captured JSON files or --fetch-readonly")

    events: list[dict[str, Any]] = [{
        **_base_event(event_type="V2_RUN_CONTEXT", event_seq=1, run_id=run_id, timestamp_ns=timestamp_ns),
        "venue_id": "lighter",
        "decision": "HOLD",
        "decision_reason_primary": "phase51b_readonly_lighter_native_limit_context",
        "decision_reason_secondary_list": ["nonlive_readonly_evidence_only", "no_order_submission", "not_live_authorization"],
        "admissible_for_financial_claim": False,
    }]
    events.append(_account_profile_event(
        event_seq=len(events) + 1,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        account_source=sources.get("account"),
        order_books_source=sources.get("order_books"),
        account_index=resolved_account_index,
        market_symbol=resolved_market_symbol,
        market_id=resolved_market_id,
    ))
    if "account_limits" in sources:
        events.append(_account_limits_event(
            event_seq=len(events) + 1,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            limits_source=sources["account_limits"],
            account_index=resolved_account_index,
        ))
    else:
        events.append({
            **_base_event(event_type="V2_GUARDRAIL_EVENT", event_seq=len(events) + 1, run_id=run_id, timestamp_ns=timestamp_ns),
            "venue_id": "lighter",
            "decision": "HOLD",
            "decision_reason_primary": "missing_lighter_account_limits_source",
            "admissible_for_financial_claim": False,
        })
    if "active_orders" in sources:
        events.append(_active_orders_event(
            event_seq=len(events) + 1,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            active_orders_source=sources["active_orders"],
            limits_source=sources.get("account_limits"),
            account_index=resolved_account_index,
            market_id=resolved_market_id,
        ))
    else:
        events.append({
            **_base_event(event_type="V2_GUARDRAIL_EVENT", event_seq=len(events) + 1, run_id=run_id, timestamp_ns=timestamp_ns),
            "venue_id": "lighter",
            "decision": "HOLD",
            "decision_reason_primary": "missing_lighter_active_orders_source",
            "admissible_for_financial_claim": False,
        })
    if "trades" in sources:
        events.append(_trades_event(
            event_seq=len(events) + 1,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            trades_source=sources["trades"],
            account_index=resolved_account_index,
            market_id=resolved_market_id,
        ))

    source_paths = _copy_sanitized_sources(out_dir, sources)
    telemetry_path = out_dir / "telemetry.jsonl"
    summary_path = out_dir / "lighter_account_native_limits_summary.json"
    gate_path = out_dir / "gate_result.json"
    command_log_path = out_dir / "command_log.json"
    spec_resolved_path = out_dir / "spec_resolved.json"
    _write_jsonl(telemetry_path, events)
    capture_complete = "account_limits" in sources and "active_orders" in sources
    summary = {
        "experiment_id": spec.get("experiment_id"),
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "venue_id": "lighter",
        "created_utc": created_utc,
        "event_count": len(events),
        "source_names": sorted(sources.keys()),
        "source_sha256": {name: source["sha256"] for name, source in sorted(sources.items())},
        "account_index_present": resolved_account_index is not None,
        "market_id_present": resolved_market_id is not None,
        "market_symbol": resolved_market_symbol,
        "phase51b_capture_complete": capture_complete,
        "gate_status": "HOLD",
        "gate_reason": "nonlive_lighter_native_limit_capture_only",
        "no_live_flag": True,
        "admissible_for_financial_claim": False,
    }
    gate = {
        "status": "HOLD",
        "reason": summary["gate_reason"],
        "phase51b_capture_complete": capture_complete,
        "approved_for_nonlive_evidence_review": capture_complete,
        "approved_for_calibration_label_ingestion": False,
        "calibration_label_ingestion_hold_reason": "requires_external_schema_validation_and_secret_audit",
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
    }
    resolved_spec = dict(spec)
    resolved_spec.update({
        "run_id": run_id,
        "spec_path": str(spec_path),
        "output_dir": str(out_dir),
        "base_url": resolved_base_url,
        "account_index_present": resolved_account_index is not None,
        "market_id": resolved_market_id,
        "market_symbol": resolved_market_symbol,
        "fetch_readonly": fetch_readonly,
        "include_trades": include_trades,
        "allow_sdk_auth": allow_sdk_auth,
        "env_file": str(env_file) if env_file else None,
        "lighter_sdk_path": str(lighter_sdk_path) if lighter_sdk_path else None,
    })
    command_log = {
        "argv": [arg if "PRIVATE" not in arg.upper() and "TOKEN" not in arg.upper() else "<redacted>" for arg in sys.argv],
        "created_utc": created_utc,
        "python_version": sys.version.split()[0],
        "fetch_readonly": fetch_readonly,
        "lighter_auth_token_present": bool(env.get("LIGHTER_AUTH_TOKEN", "").strip()),
        "lighter_api_private_key_present": bool(env.get("LIGHTER_API_PRIVATE_KEY_HEX", "").strip()),
    }
    _write_json(summary_path, summary)
    _write_json(gate_path, gate)
    _write_json(spec_resolved_path, _redact(resolved_spec))
    _write_json(command_log_path, command_log)
    artifact_paths = [telemetry_path, summary_path, gate_path, command_log_path, spec_resolved_path, *source_paths]
    metadata = {
        "experiment_id": spec.get("experiment_id"),
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "venue_id": "lighter",
        "no_live_flag": True,
        "capital_escalation_flag": False,
        "risk_limit_override_flag": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "phase51b_capture_complete": capture_complete,
        "source_sha256": summary["source_sha256"],
    }
    evidence_index = _write_evidence_index(out_dir, artifact_paths, metadata)
    artifact_paths.append(evidence_index)
    _write_manifest(out_dir, artifact_paths, metadata, created_utc)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/phase51b_lighter_account_native_limits.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--account-json", type=Path, default=None)
    parser.add_argument("--account-limits-json", type=Path, default=None)
    parser.add_argument("--active-orders-json", type=Path, default=None)
    parser.add_argument("--order-books-json", type=Path, default=None)
    parser.add_argument("--trades-json", type=Path, default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional KEY=VALUE env file loaded without shell execution.",
    )
    parser.add_argument("--fetch-readonly", action="store_true")
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--allow-sdk-auth", action="store_true")
    parser.add_argument(
        "--lighter-sdk-path",
        type=Path,
        default=None,
        help="Optional explicit local lighter-sdk source path used only when the package is not installed.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--account-index", type=int, default=None)
    parser.add_argument("--market-id", type=int, default=None)
    parser.add_argument("--market-symbol", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()
    try:
        out_dir = run(
            spec_path=args.spec,
            output_root=args.output_root,
            run_id=args.run_id,
            account_json=args.account_json,
            account_limits_json=args.account_limits_json,
            active_orders_json=args.active_orders_json,
            order_books_json=args.order_books_json,
            trades_json=args.trades_json,
            env_file=args.env_file,
            fetch_readonly=args.fetch_readonly,
            include_trades=args.include_trades,
            allow_sdk_auth=args.allow_sdk_auth,
            lighter_sdk_path=args.lighter_sdk_path,
            base_url=args.base_url,
            account_index=args.account_index,
            market_id=args.market_id,
            market_symbol=args.market_symbol,
            timestamp_ns=args.timestamp_ns,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"phase51b_lighter_account_limits: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51b_lighter_account_limits: wrote {out_dir}")
    print("phase51b_lighter_account_limits: status HOLD (read-only nonlive evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
