#!/usr/bin/env python3
"""SDK-backed bridge for Extended account/execution actions.

The Rust connector invokes this helper via:
  PARAPHINA_EXTENDED_BRIDGE_OP=<snapshot|place|cancel|cancel_all>
  PARAPHINA_EXTENDED_BRIDGE_PAYLOAD='{"...": "..."}'

It can also be run manually with CLI subcommands for debugging.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from decimal import Decimal
from typing import Any


def fatal(message: str, exit_code: int = 1) -> "NoReturn":
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        fatal(f"missing required environment variable: {name}")
    return value


def load_sdk():
    try:
        from x10.perpetual.accounts import StarkPerpetualAccount
        from x10.perpetual.configuration import MAINNET_CONFIG, TESTNET_CONFIG
        from x10.perpetual.orders import OrderSide, TimeInForce
        from x10.perpetual.trading_client import PerpetualTradingClient
    except ImportError as exc:
        fatal(
            "extended bridge requires the official SDK: "
            "`python3 -m pip install x10-python-trading-starknet` "
            f"(import error: {exc})"
        )
    return StarkPerpetualAccount, MAINNET_CONFIG, TESTNET_CONFIG, OrderSide, TimeInForce, PerpetualTradingClient


def load_endpoint_config():
    _, MAINNET_CONFIG, TESTNET_CONFIG, _, _, _ = load_sdk()
    env = os.getenv("EXTENDED_SDK_ENV", "mainnet").strip().lower()
    if env in {"testnet", "sepolia"}:
        return TESTNET_CONFIG
    return MAINNET_CONFIG


def build_account():
    StarkPerpetualAccount, _, _, _, _, _ = load_sdk()
    api_key = require_env("EXTENDED_API_KEY")
    private_key = require_env("EXTENDED_STARK_PRIVATE_KEY")
    public_key = require_env("EXTENDED_STARK_PUBLIC_KEY")
    vault = require_env("EXTENDED_L2_VAULT")
    return StarkPerpetualAccount(
        vault=vault,
        private_key=private_key,
        public_key=public_key,
        api_key=api_key,
    )


def to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def bool_value(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def require_payload(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        fatal(f"missing required payload field: {key}")
    return payload[key]


async def build_client():
    _, _, _, _, _, PerpetualTradingClient = load_sdk()
    return PerpetualTradingClient(
        endpoint_config=load_endpoint_config(),
        stark_account=build_account(),
    )


async def op_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    market = str(require_payload(payload, "market")).strip()
    client = await build_client()
    try:
        balance_resp = await client.account.get_balance()
        if balance_resp.data is None:
            fatal("extended get_balance returned no data")
        positions_resp = await client.account.get_positions(
            market_names=[market] if market else None
        )
        positions = list(positions_resp.data or [])
        timestamp_ms = balance_resp.data.updated_time
        if positions:
            timestamp_ms = max(
                timestamp_ms,
                max(position.updated_at for position in positions),
            )
        normalized_positions = []
        for position in positions:
            signed_size = to_float(position.size)
            if str(position.side).upper() == "SHORT":
                signed_size = -signed_size
            normalized_positions.append(
                {
                    "market": position.market,
                    "size": signed_size,
                    "entry_price": to_float(position.open_price),
                    "liquidation_price": (
                        None
                        if position.liquidation_price is None
                        else to_float(position.liquidation_price)
                    ),
                    "updated_at": position.updated_at,
                }
            )
        return {
            "timestamp_ms": timestamp_ms,
            "collateral_asset": balance_resp.data.collateral_name,
            "balance_usd": to_float(balance_resp.data.balance),
            "used_usd": to_float(balance_resp.data.initial_margin),
            "available_usd": to_float(balance_resp.data.available_for_trade),
            "positions": normalized_positions,
        }
    finally:
        await client.close()


async def op_place(payload: dict[str, Any]) -> dict[str, Any]:
    _, _, _, OrderSide, TimeInForce, _ = load_sdk()
    market = str(require_payload(payload, "market")).strip()
    side = OrderSide(str(require_payload(payload, "side")).upper())
    client_order_id = str(require_payload(payload, "client_order_id")).strip()
    time_in_force = TimeInForce(str(payload.get("time_in_force", "GTT")).upper())
    client = await build_client()
    try:
        response = await client.place_order(
            market_name=market,
            amount_of_synthetic=Decimal(str(require_payload(payload, "size"))),
            price=Decimal(str(require_payload(payload, "price"))),
            side=side,
            post_only=bool_value(payload, "post_only"),
            reduce_only=bool_value(payload, "reduce_only"),
            time_in_force=time_in_force,
            external_id=client_order_id,
        )
        if response.data is None:
            fatal("extended place_order returned no data")
        return {
            "order_id": str(response.data.id),
            "client_order_id": response.data.external_id,
        }
    finally:
        await client.close()


async def op_cancel(payload: dict[str, Any]) -> dict[str, Any]:
    raw_order_id = str(require_payload(payload, "order_id")).strip()
    client = await build_client()
    try:
        if raw_order_id.isdigit():
            await client.orders.cancel_order(order_id=int(raw_order_id))
        else:
            await client.orders.cancel_order_by_external_id(raw_order_id)
        return {"order_id": raw_order_id}
    finally:
        await client.close()


async def op_cancel_all(payload: dict[str, Any]) -> dict[str, Any]:
    market = str(require_payload(payload, "market")).strip()
    client = await build_client()
    try:
        open_orders_resp = await client.account.get_open_orders(
            market_names=[market] if market else None
        )
        open_orders = list(open_orders_resp.data or [])
        order_ids = [order.id for order in open_orders]
        if order_ids:
            await client.orders.mass_cancel(order_ids=order_ids)
        return {"count": len(order_ids)}
    finally:
        await client.close()


def parse_payload() -> tuple[str, dict[str, Any]]:
    op = os.getenv("PARAPHINA_EXTENDED_BRIDGE_OP", "").strip()
    payload_raw = os.getenv("PARAPHINA_EXTENDED_BRIDGE_PAYLOAD", "").strip()
    if op:
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except json.JSONDecodeError as exc:
            fatal(f"invalid PARAPHINA_EXTENDED_BRIDGE_PAYLOAD: {exc}")
        if not isinstance(payload, dict):
            fatal("extended bridge payload must decode to a JSON object")
        return op, payload

    parser = argparse.ArgumentParser(description="Extended SDK bridge")
    sub = parser.add_subparsers(dest="op", required=True)

    snapshot = sub.add_parser("snapshot", help="fetch normalized account snapshot")
    snapshot.add_argument("--market", required=True)

    place = sub.add_parser("place", help="place a limit order")
    place.add_argument("--market", required=True)
    place.add_argument("--side", required=True, choices=["BUY", "SELL"])
    place.add_argument("--price", required=True)
    place.add_argument("--size", required=True)
    place.add_argument("--client-order-id", required=True)
    place.add_argument("--time-in-force", default="GTT")
    place.add_argument("--post-only", action="store_true")
    place.add_argument("--reduce-only", action="store_true")

    cancel = sub.add_parser("cancel", help="cancel an order")
    cancel.add_argument("--order-id", required=True)

    cancel_all = sub.add_parser("cancel_all", help="cancel all orders for a market")
    cancel_all.add_argument("--market", required=True)

    args = parser.parse_args()
    payload = {
        key.replace("_", "-"): value
        for key, value in vars(args).items()
        if key != "op" and value is not None
    }
    normalized = {
        "market": payload.get("market"),
        "side": payload.get("side"),
        "price": payload.get("price"),
        "size": payload.get("size"),
        "client_order_id": payload.get("client-order-id"),
        "time_in_force": payload.get("time-in-force"),
        "post_only": payload.get("post-only", False),
        "reduce_only": payload.get("reduce-only", False),
        "order_id": payload.get("order-id"),
    }
    return args.op, {k: v for k, v in normalized.items() if v is not None}


async def main_async() -> None:
    if sys.version_info < (3, 10):
        fatal("extended bridge requires Python 3.10+")
    op, payload = parse_payload()
    handlers = {
        "snapshot": op_snapshot,
        "place": op_place,
        "cancel": op_cancel,
        "cancel_all": op_cancel_all,
    }
    handler = handlers.get(op)
    if handler is None:
        fatal(f"unsupported extended bridge op: {op}")
    result = await handler(payload)
    print(json.dumps(result, separators=(",", ":")))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
