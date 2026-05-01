#!/usr/bin/env python3
"""Local Lighter signer bridge for Paraphina.

Reads Lighter API credentials from the environment and exposes a small local
HTTP API compatible with `LighterSignerClient` in the Rust connector:

- POST /sign
- GET /health
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests


def _patch_lighter_sdk_platform_detection() -> None:
    """Work around lighter-sdk treating Linux/aarch64 as unsupported.

    The package ships `lighter-signer-linux-arm64.so`, but its own platform
    check only accepts `platform.machine() == "arm64"` on Linux.
    """
    machine = platform.machine().lower()
    if platform.system() == "Linux" and machine == "aarch64":
        platform.machine = lambda: "arm64"  # type: ignore[assignment]


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing required env: {name}")
    return value


def _normalize_hex(value: str) -> str:
    return value[2:] if value.startswith("0x") else value


def _normalize_market_symbol(symbol: str) -> str:
    upper = symbol.strip().upper()
    for suffix in ("-USD-PERP", "-PERP", "-USD"):
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper


def _resolve_base_url() -> str:
    explicit = os.environ.get("LIGHTER_REST_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    network = os.environ.get("LIGHTER_NETWORK", "mainnet").strip().lower()
    if network == "mainnet":
        return "https://mainnet.zklighter.elliot.ai"
    if network == "testnet":
        return "https://testnet.zklighter.elliot.ai"
    raise SystemExit("unsupported LIGHTER_NETWORK value; use mainnet or testnet")


class LighterSignerBridge:
    def __init__(self) -> None:
        _patch_lighter_sdk_platform_detection()
        try:
            from lighter import SignerClient
        except ImportError as exc:  # pragma: no cover - runtime only
            raise SystemExit(
                "lighter-sdk is not installed. Install it with: pip install lighter-sdk"
            ) from exc

        self.base_url = _resolve_base_url()
        self.account_index = int(_require("LIGHTER_ACCOUNT_INDEX"))
        self.api_key_index = int(_require("LIGHTER_API_KEY_INDEX"))
        self.market_symbol = os.environ.get("LIGHTER_MARKET", "ETH-USD").strip() or "ETH-USD"
        self.market_id_env = os.environ.get("LIGHTER_MARKET_ID", "").strip()
        self.client = asyncio.run(
            self._build_signer_client(
                SignerClient,
                self.base_url,
                self.account_index,
                {self.api_key_index: _normalize_hex(_require("LIGHTER_API_PRIVATE_KEY_HEX"))},
            )
        )
        self._market_index: int | None = (
            int(self.market_id_env) if self.market_id_env else None
        )
        self._market_lock = threading.Lock()

    @staticmethod
    async def _build_signer_client(
        signer_client_cls: Any,
        base_url: str,
        account_index: int,
        api_private_keys: dict[int, str],
    ) -> Any:
        return signer_client_cls(base_url, account_index, api_private_keys)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "base_url": self.base_url,
            "account_index": self.account_index,
            "api_key_index": self.api_key_index,
            "market": self.market_symbol,
            "market_index": self.market_index(),
        }

    def market_index(self) -> int:
        if self._market_index is not None:
            return self._market_index
        with self._market_lock:
            if self._market_index is not None:
                return self._market_index
            url = f"{self.base_url}/api/v1/orderBooks"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            books = payload.get("order_books") or payload.get("orderBooks") or payload
            if not isinstance(books, list):
                raise ValueError("unexpected orderBooks payload from Lighter")
            target = _normalize_market_symbol(self.market_symbol)
            for book in books:
                symbol = book.get("symbol")
                market_id = book.get("market_id", book.get("marketId"))
                if not symbol or market_id is None:
                    continue
                if _normalize_market_symbol(str(symbol)) == target:
                    self._market_index = int(market_id)
                    return self._market_index
            raise ValueError(f"LIGHTER_MARKET not found in orderBooks response: {self.market_symbol}")

    def sign(self, payload: dict[str, Any]) -> dict[str, Any]:
        op = str(payload.get("op", "")).strip().lower()
        if op == "create_order":
            return self._sign_create_order(payload)
        if op == "cancel_order":
            return self._sign_cancel_order(payload)
        if op == "modify_order":
            return self._sign_modify_order(payload)
        if op == "cancel_all":
            return self._sign_cancel_all(payload)
        raise ValueError(f"unsupported op: {op!r}")

    def _decode_signed(
        self, result: tuple[Any, Any, Any, Any]
    ) -> dict[str, Any]:
        tx_type, tx_info, tx_hash, err = result
        if err:
            raise ValueError(str(err))
        parsed_info = json.loads(tx_info) if isinstance(tx_info, str) else tx_info
        return {
            "tx_type": int(tx_type),
            "tx_info": parsed_info,
            "tx_hash": tx_hash,
        }

    def _map_order_type(self, raw: str) -> int:
        order_type = raw.strip().lower()
        if order_type == "limit":
            return self.client.ORDER_TYPE_LIMIT
        if order_type == "market":
            return self.client.ORDER_TYPE_MARKET
        raise ValueError(f"unsupported Lighter order_type: {raw!r}")

    def _map_time_in_force(self, raw: str, post_only: bool) -> int:
        if post_only:
            return self.client.ORDER_TIME_IN_FORCE_POST_ONLY
        tif = raw.strip().lower()
        if tif == "gtc":
            return self.client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME
        if tif == "ioc":
            return self.client.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL
        raise ValueError(f"unsupported Lighter time_in_force: {raw!r}")

    def _sign_create_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        post_only = bool(int(payload.get("post_only", 0)))
        trigger_price = payload.get("trigger_price")
        order_expiry = payload.get("order_expiry")
        result = self.client.sign_create_order(
            market_index=int(payload.get("market_index", self.market_index())),
            client_order_index=int(payload["client_order_index"]),
            base_amount=int(payload["base_amount"]),
            price=int(payload["price"]),
            is_ask=bool(int(payload["is_ask"])),
            order_type=self._map_order_type(str(payload.get("order_type", "limit"))),
            time_in_force=self._map_time_in_force(
                str(payload.get("time_in_force", "Gtc")),
                post_only=post_only,
            ),
            reduce_only=bool(int(payload.get("reduce_only", 0))),
            trigger_price=(
                int(trigger_price)
                if trigger_price is not None
                else self.client.NIL_TRIGGER_PRICE
            ),
            order_expiry=(
                int(order_expiry)
                if order_expiry is not None
                else self.client.DEFAULT_28_DAY_ORDER_EXPIRY
            ),
            nonce=int(payload["nonce"]),
            api_key_index=self.api_key_index,
        )
        return self._decode_signed(result)

    def _sign_cancel_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        order_index = payload.get("order_index")
        if order_index is None:
            order_index = payload.get("client_order_index")
        if order_index is None:
            raise ValueError("cancel_order requires order_index or client_order_index")
        result = self.client.sign_cancel_order(
            market_index=int(payload.get("market_index", self.market_index())),
            order_index=int(order_index),
            nonce=int(payload["nonce"]),
            api_key_index=self.api_key_index,
        )
        return self._decode_signed(result)

    def _sign_modify_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        order_index = payload.get("order_index")
        if order_index is None:
            order_index = payload.get("client_order_index")
        if order_index is None:
            raise ValueError("modify_order requires order_index or client_order_index")
        trigger_price = payload.get("trigger_price")
        result = self.client.sign_modify_order(
            market_index=int(payload.get("market_index", self.market_index())),
            order_index=int(order_index),
            base_amount=int(payload["base_amount"]),
            price=int(payload["price"]),
            trigger_price=(
                int(trigger_price)
                if trigger_price is not None
                else self.client.NIL_TRIGGER_PRICE
            ),
            nonce=int(payload["nonce"]),
            api_key_index=self.api_key_index,
        )
        return self._decode_signed(result)

    def _sign_cancel_all(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.client.sign_cancel_all_orders(
            time_in_force=int(payload["cancel_all_time_in_force"]),
            timestamp_ms=int(payload["cancel_all_time"]),
            nonce=int(payload["nonce"]),
            api_key_index=self.api_key_index,
        )
        return self._decode_signed(result)


class _Handler(BaseHTTPRequestHandler):
    bridge: LighterSignerBridge

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            self._send_json(200, self.bridge.health())
        except Exception as exc:  # pragma: no cover - runtime only
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/sign":
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8"))
            self._send_json(200, self.bridge.sign(payload))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - runtime only
            self._send_json(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write(f"lighter_signer_service | {fmt % args}\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local Lighter signer bridge for Paraphina."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=9001, help="Bind port.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bridge = LighterSignerBridge()
    _Handler.bridge = bridge
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"lighter_signer_service listening on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
