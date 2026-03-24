#!/usr/bin/env python3
"""Sign Paradex order payloads for Paraphina live order submission."""

from __future__ import annotations

import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path

VALID_ENVS = ("prod", "testnet", "nightly")
CACHE_TTL_SECS = 24 * 60 * 60


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing required env: {name}")
    return value


def _load_payload() -> dict[str, object]:
    raw = os.environ.get("PARADEX_ORDER_PAYLOAD", "").strip()
    if not raw:
        raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit("missing PARADEX_ORDER_PAYLOAD")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("PARADEX_ORDER_PAYLOAD must decode to a JSON object")
    return payload


def _cache_path(env_name: str) -> Path:
    override = os.environ.get("PARADEX_SYSTEM_CONFIG_CACHE", "").strip()
    if override:
        return Path(override)
    return Path(f"/tmp/paradex_system_config_{env_name}.json")


def _load_system_config(env_name: str):
    from paradex_py.api.api_client import ParadexApiClient
    from paradex_py.api.models import SystemConfigSchema

    cache_path = _cache_path(env_name)
    if cache_path.exists():
        age_secs = time.time() - cache_path.stat().st_mtime
        if age_secs < CACHE_TTL_SECS:
            return SystemConfigSchema().loads(cache_path.read_text())

    config = ParadexApiClient(env=env_name).fetch_system_config()
    try:
        cache_path.write_text(SystemConfigSchema().dumps(config))
    except OSError:
        pass
    return config


def _build_order(payload: dict[str, object], signature_timestamp: int):
    from paradex_py.common.order import Order, OrderSide, OrderType

    side_raw = str(payload["side"]).upper()
    if side_raw == "BUY":
        side = OrderSide.Buy
    elif side_raw == "SELL":
        side = OrderSide.Sell
    else:
        raise SystemExit(f"unsupported Paradex side: {side_raw!r}")

    order_type = OrderType(str(payload.get("type", "LIMIT")).upper())
    flags = payload.get("flags") or []
    reduce_only = any(str(flag).upper() == "REDUCE_ONLY" for flag in flags)
    trigger_price_raw = payload.get("trigger_price")

    return Order(
        market=str(payload["market"]),
        order_type=order_type,
        order_side=side,
        size=Decimal(str(payload["size"])),
        limit_price=Decimal(str(payload.get("price", "0"))),
        client_id=str(payload.get("client_id", "")),
        signature_timestamp=signature_timestamp,
        instruction=str(payload.get("instruction", "GTC")).upper(),
        reduce_only=reduce_only,
        recv_window=(
            int(payload["recv_window"]) if payload.get("recv_window") is not None else None
        ),
        stp=str(payload["stp"]) if payload.get("stp") is not None else None,
        trigger_price=(
            Decimal(str(trigger_price_raw)) if trigger_price_raw is not None else None
        ),
    )


def main() -> int:
    env_name = os.environ.get("PARADEX_PY_ENV", "prod").strip().lower() or "prod"
    if env_name not in VALID_ENVS:
        print(f"invalid PARADEX_PY_ENV: {env_name!r}", file=sys.stderr)
        return 2

    try:
        from paradex_py.account.subkey_account import SubkeyAccount
    except ImportError:
        print(
            "paradex_py is not installed. Install it with: python3 -m pip install paradex_py",
            file=sys.stderr,
        )
        return 2

    try:
        payload = _load_payload()
        config = _load_system_config(env_name)
        account = SubkeyAccount(
            config=config,
            l2_private_key=_require("PARADEX_L2_PRIVATE_KEY"),
            l2_address=_require("PARADEX_L2_ADDRESS"),
        )
        signature_timestamp = int(payload.get("signature_timestamp") or int(time.time() * 1000))
        order = _build_order(payload, signature_timestamp)
        order.signature = account.sign_order(order)
        signed_payload = order.dump_to_dict()
        if payload.get("on_behalf_of_account") is not None:
            signed_payload["on_behalf_of_account"] = payload["on_behalf_of_account"]
        if payload.get("vwap_price") is not None:
            signed_payload["vwap_price"] = str(payload["vwap_price"])
        print(json.dumps(signed_payload, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
