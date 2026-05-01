#!/usr/bin/env python3
"""Read-only Aster fee-path attribution gate for the current Phase 5 surface."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from phase5_econ_attribution import (  # noqa: E402
    ASTER_DOCS,
    DEFAULT_ENV_FILE,
    decimal_of,
    decimal_to_float,
    http_json,
    load_env,
    load_json,
    sign_aster_query,
    write_json,
    write_text,
)

UTC = timezone.utc
ZERO = Decimal("0")
DEFAULT_RUN_ROOT = Path(
    "/home/ubuntu/promotion_runs/"
    "phase5_all5_current_surface_lighter_post_replace_2h_pnl_observation_clearance_7200s_20260423T215842Z/"
    "live_canary"
)
DEFAULT_OUT_DIR = ROOT / "phase5" / "runs" / "phase5_all5_current_surface_aster_fee_path_attribution_gate"
DEFAULT_REPORT_PATH = (
    ROOT
    / "docs"
    / "INVESTIGATIONS"
    / f"phase5_current_surface_aster_fee_path_attribution_{datetime.now(UTC).strftime('%Y%m%d')}.md"
)

CATEGORY_MAKER_MM = "maker_mm_post_only"
CATEGORY_TAKER_UNWIND = "taker_reduce_only_unwind"
CATEGORY_CLEANUP = "cleanup_or_guard_flatten"
CATEGORY_UNKNOWN = "unknown_order_path"
CATEGORY_UNMATCHED = "unmatched_venue_trade"


@dataclass
class TelemetryTruth:
    fills_by_order_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    fills_by_client_order_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders_by_order_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders_by_client_order_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    aster_fill_count: int = 0
    aster_fill_base: Decimal = ZERO
    aster_mm_fill_count: int = 0
    aster_hedge_fill_count: int = 0
    aster_order_event_count: int = 0


@dataclass
class JoinedTrade:
    trade_id: str
    order_id: str
    client_order_id: str
    time_ms: int
    side: str
    qty: Decimal
    price: Decimal
    quote_qty: Decimal
    commission: Decimal
    commission_asset: str
    realized_pnl: Decimal
    maker: bool | None
    time_in_force: str
    reduce_only: bool | None
    post_only: bool | None
    purpose: str
    telemetry_matched: bool
    order_matched: bool
    category: str
    notes: list[str] = field(default_factory=list)

    def notional(self) -> Decimal:
        if self.quote_qty != ZERO:
            return abs(self.quote_qty)
        return abs(self.qty * self.price)

    def fee_bps(self) -> Decimal | None:
        notional = self.notional()
        if notional == ZERO:
            return None
        return abs(self.commission) / notional * Decimal("10000")

    def to_dict(self) -> dict[str, Any]:
        fee_bps = self.fee_bps()
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "time_ms": self.time_ms,
            "side": self.side,
            "qty": decimal_to_float(self.qty),
            "price": decimal_to_float(self.price),
            "quote_qty": decimal_to_float(self.quote_qty),
            "notional_usd": decimal_to_float(self.notional()),
            "commission_usd": decimal_to_float(self.commission),
            "commission_asset": self.commission_asset,
            "realized_pnl_usd": decimal_to_float(self.realized_pnl),
            "fee_bps": None if fee_bps is None else decimal_to_float(fee_bps),
            "maker": self.maker,
            "time_in_force": self.time_in_force,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
            "purpose": self.purpose,
            "telemetry_matched": self.telemetry_matched,
            "order_matched": self.order_matched,
            "category": self.category,
            "notes": list(self.notes),
        }


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def string_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with open_maybe_gzip(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            cursor = 0
            while cursor < len(line):
                while cursor < len(line) and line[cursor].isspace():
                    cursor += 1
                if cursor >= len(line):
                    break
                try:
                    payload, end = decoder.raw_decode(line, cursor)
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(
                        f"{path}:{line_no}: {exc.msg}",
                        exc.doc,
                        exc.pos,
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(payload).__name__}")
                yield payload
                cursor = end


def find_telemetry_path(run_root: Path) -> Path | None:
    for name in ("telemetry_bounded.jsonl", "telemetry_bounded.jsonl.gz", "telemetry.jsonl", "telemetry.jsonl.gz"):
        path = run_root / name
        if path.exists():
            return path
    return None


def load_run_window(run_root: Path, start_ms: int | None, end_ms: int | None) -> tuple[int, int]:
    if start_ms is not None and end_ms is not None:
        return start_ms, end_ms
    summary_path = run_root / "live_segment_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        first_ms = int(summary.get("first_ts_ms") or 0)
        last_ms = int(summary.get("last_ts_ms") or 0)
        if first_ms > 0 and last_ms >= first_ms:
            return start_ms or first_ms, end_ms or last_ms
    raise RuntimeError("run window unavailable; pass --start-ms and --end-ms")


def collect_aster_telemetry(run_root: Path) -> tuple[TelemetryTruth, dict[str, Any]]:
    truth = TelemetryTruth()
    telemetry_path = find_telemetry_path(run_root)
    if telemetry_path is None:
        return truth, {"telemetry_path": None, "telemetry_available": False}

    first_tick: int | None = None
    last_tick: int | None = None
    rows = 0
    for row in iter_jsonl(telemetry_path):
        rows += 1
        tick = row.get("t")
        if isinstance(tick, int):
            first_tick = tick if first_tick is None else min(first_tick, tick)
            last_tick = tick if last_tick is None else max(last_tick, tick)
        for order in row.get("orders") or []:
            if not isinstance(order, dict) or str(order.get("venue_id") or "").lower() != "aster":
                continue
            enriched = dict(order)
            enriched["telemetry_tick"] = tick
            truth.aster_order_event_count += 1
            order_id = string_id(enriched.get("order_id"))
            client_order_id = string_id(enriched.get("client_order_id"))
            if order_id:
                truth.orders_by_order_id[order_id] = enriched
            if client_order_id:
                truth.orders_by_client_order_id[client_order_id] = enriched
        for fill in row.get("fills") or []:
            if not isinstance(fill, dict) or str(fill.get("venue_id") or "").lower() != "aster":
                continue
            enriched = dict(fill)
            enriched["telemetry_tick"] = tick
            truth.aster_fill_count += 1
            truth.aster_fill_base += abs(decimal_of(enriched.get("size")))
            purpose = str(enriched.get("purpose") or "").lower()
            if purpose == "mm":
                truth.aster_mm_fill_count += 1
            elif purpose:
                truth.aster_hedge_fill_count += 1
            order_id = string_id(enriched.get("order_id"))
            client_order_id = string_id(enriched.get("client_order_id"))
            if order_id:
                truth.fills_by_order_id[order_id] = enriched
            if client_order_id:
                truth.fills_by_client_order_id[client_order_id] = enriched

    meta = {
        "telemetry_path": str(telemetry_path),
        "telemetry_available": True,
        "rows": rows,
        "first_tick": first_tick,
        "last_tick": last_tick,
    }
    return truth, meta


def aster_signed_get(env: dict[str, str], path: str, params: dict[str, Any]) -> Any:
    rest_url = env.get("ASTER_REST_URL", "https://fapi.asterdex.com").rstrip("/")
    api_key = env["ASTER_API_KEY"]
    query = sign_aster_query(env, {key: value for key, value in params.items() if value is not None})
    return http_json("GET", f"{rest_url}{path}?{query}", headers={"X-MBX-APIKEY": api_key})


def aster_paginated(
    env: dict[str, str],
    path: str,
    *,
    start_ms: int,
    end_ms: int,
    symbol: str | None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params: dict[str, Any] = {"startTime": cursor, "endTime": end_ms, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        payload = aster_signed_get(env, path, params)
        if not payload:
            break
        batch = [item for item in payload if isinstance(item, dict)]
        items.extend(batch)
        batch_times = [int(item.get("time") or item.get("updateTime") or 0) for item in batch]
        batch_max = max(batch_times or [cursor])
        if len(batch) < limit or batch_max < cursor:
            break
        next_cursor = batch_max + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    return items


def fetch_aster_venue_truth(env: dict[str, str], *, start_ms: int, end_ms: int) -> dict[str, Any]:
    symbol = env.get("ASTER_MARKET", "ETHUSDT").strip() or "ETHUSDT"
    income = aster_paginated(env, "/fapi/v1/income", start_ms=start_ms, end_ms=end_ms, symbol=symbol)
    trades = aster_paginated(env, "/fapi/v1/userTrades", start_ms=start_ms, end_ms=end_ms, symbol=symbol)
    orders = aster_paginated(env, "/fapi/v1/allOrders", start_ms=start_ms, end_ms=end_ms, symbol=symbol)
    try:
        commission_rate = aster_signed_get(env, "/fapi/v1/commissionRate", {"symbol": symbol})
    except Exception as exc:  # pragma: no cover - live API fallback
        commission_rate = {"error": str(exc)}
    return {
        "symbol": symbol,
        "income": income,
        "trades": trades,
        "orders": orders,
        "commission_rate": commission_rate,
    }


def index_orders(orders: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_order_id: dict[str, dict[str, Any]] = {}
    by_client_id: dict[str, dict[str, Any]] = {}
    for order in orders:
        order_id = string_id(first_present(order, "orderId", "order_id"))
        client_id = string_id(first_present(order, "clientOrderId", "client_order_id", "origClientOrderId"))
        if order_id:
            by_order_id[order_id] = order
        if client_id:
            by_client_id[client_id] = order
    return by_order_id, by_client_id


def choose_joined_payload(
    trade: dict[str, Any],
    order_by_id: dict[str, dict[str, Any]],
    order_by_client: dict[str, dict[str, Any]],
    telemetry: TelemetryTruth,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    order_id = string_id(first_present(trade, "orderId", "order_id"))
    client_id = string_id(first_present(trade, "clientOrderId", "client_order_id"))
    order = order_by_id.get(order_id) if order_id else None
    if order is None and client_id:
        order = order_by_client.get(client_id)
    if order is not None and not client_id:
        client_id = string_id(first_present(order, "clientOrderId", "client_order_id", "origClientOrderId"))

    telemetry_fill = telemetry.fills_by_order_id.get(order_id) if order_id else None
    if telemetry_fill is None and client_id:
        telemetry_fill = telemetry.fills_by_client_order_id.get(client_id)
    return order, telemetry_fill


def classify_trade(
    *,
    maker: bool | None,
    purpose: str,
    time_in_force: str,
    reduce_only: bool | None,
    post_only: bool | None,
    telemetry_matched: bool,
    order_matched: bool,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    purpose_l = purpose.strip().lower()
    tif = time_in_force.strip().upper()

    if not telemetry_matched and not order_matched:
        return CATEGORY_UNMATCHED, ["no internal telemetry or Aster order-history match"]

    if purpose_l in {"canary", "guard", "cleanup", "closeout", "exit"}:
        return CATEGORY_CLEANUP, ["guard/cleanup purpose"]

    if reduce_only is True:
        return CATEGORY_TAKER_UNWIND, ["reduceOnly order path"]

    if purpose_l in {"hedge", "unwind", "soft_unwind", "emergency"}:
        return CATEGORY_TAKER_UNWIND, ["non-MM unwind/hedge purpose"]

    if purpose_l == "mm" and post_only is True and maker is True:
        return CATEGORY_MAKER_MM, ["MM post-only maker fill"]

    if purpose_l == "mm" and tif == "GTX" and maker is True:
        return CATEGORY_MAKER_MM, ["Aster GTX maker fill"]

    if purpose_l == "mm" and (post_only is True or tif == "GTX") and maker is False:
        return CATEGORY_UNKNOWN, ["MM post-only/GTX path reported as taker"]

    if maker is False:
        return CATEGORY_TAKER_UNWIND, ["taker fill outside confirmed MM maker path"]

    if maker is True and purpose_l == "mm":
        return CATEGORY_MAKER_MM, ["MM maker fill; post-only flag inferred incomplete"]

    notes.append("insufficient order-purpose or maker/taker evidence")
    return CATEGORY_UNKNOWN, notes


def join_trades(
    trades: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    telemetry: TelemetryTruth,
) -> list[JoinedTrade]:
    order_by_id, order_by_client = index_orders(orders)
    joined: list[JoinedTrade] = []
    for trade in trades:
        order, telemetry_fill = choose_joined_payload(trade, order_by_id, order_by_client, telemetry)
        order_id = string_id(first_present(trade, "orderId", "order_id"))
        client_id = string_id(first_present(trade, "clientOrderId", "client_order_id"))
        if order is not None and not client_id:
            client_id = string_id(first_present(order, "clientOrderId", "client_order_id", "origClientOrderId"))
        if telemetry_fill is not None and not client_id:
            client_id = string_id(telemetry_fill.get("client_order_id"))

        maker = parse_bool(first_present(trade, "maker", "isMaker"))
        tif = str(first_present(order or {}, "timeInForce", "time_in_force") or "")
        reduce_only = parse_bool(first_present(order or {}, "reduceOnly", "reduce_only"))
        post_only = None
        if tif.upper() == "GTX":
            post_only = True
        if telemetry_fill is not None:
            purpose = str(telemetry_fill.get("purpose") or "")
        else:
            purpose = ""
        telemetry_order = None
        if client_id:
            telemetry_order = telemetry.orders_by_client_order_id.get(client_id)
        if telemetry_order is None and order_id:
            telemetry_order = telemetry.orders_by_order_id.get(order_id)
        if telemetry_order is not None:
            reduce_only = parse_bool(telemetry_order.get("reduce_only")) if reduce_only is None else reduce_only
            telemetry_post_only = parse_bool(telemetry_order.get("post_only"))
            post_only = telemetry_post_only if telemetry_post_only is not None else post_only
            if not purpose:
                purpose = str(telemetry_order.get("purpose") or "")

        category, notes = classify_trade(
            maker=maker,
            purpose=purpose,
            time_in_force=tif,
            reduce_only=reduce_only,
            post_only=post_only,
            telemetry_matched=telemetry_fill is not None or telemetry_order is not None,
            order_matched=order is not None,
        )
        joined.append(
            JoinedTrade(
                trade_id=string_id(first_present(trade, "id", "tradeId", "trade_id")),
                order_id=order_id,
                client_order_id=client_id,
                time_ms=int(first_present(trade, "time", "timestamp") or 0),
                side=str(first_present(trade, "side", "buyer") or ""),
                qty=decimal_of(first_present(trade, "qty", "quantity", "executedQty")),
                price=decimal_of(first_present(trade, "price", "avgPrice")),
                quote_qty=decimal_of(first_present(trade, "quoteQty", "quoteQuantity", "notional")),
                commission=decimal_of(first_present(trade, "commission", "fee")),
                commission_asset=str(first_present(trade, "commissionAsset", "feeAsset") or ""),
                realized_pnl=decimal_of(first_present(trade, "realizedPnl", "realizedPnL", "realisedPnl")),
                maker=maker,
                time_in_force=tif,
                reduce_only=reduce_only,
                post_only=post_only,
                purpose=purpose,
                telemetry_matched=telemetry_fill is not None or telemetry_order is not None,
                order_matched=order is not None,
                category=category,
                notes=notes,
            )
        )
    return joined


def empty_category_summary() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "base_qty": ZERO,
        "notional_usd": ZERO,
        "commission_usd": ZERO,
        "realized_pnl_usd": ZERO,
        "maker_count": 0,
        "taker_count": 0,
        "unknown_maker_count": 0,
    }


def summarize_joined(
    joined: list[JoinedTrade],
    income: list[dict[str, Any]],
    telemetry: TelemetryTruth,
    telemetry_meta: dict[str, Any],
    *,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    categories = {
        CATEGORY_MAKER_MM: empty_category_summary(),
        CATEGORY_TAKER_UNWIND: empty_category_summary(),
        CATEGORY_CLEANUP: empty_category_summary(),
        CATEGORY_UNKNOWN: empty_category_summary(),
        CATEGORY_UNMATCHED: empty_category_summary(),
    }
    total = empty_category_summary()
    matched_count = 0
    order_matched_count = 0
    for trade in joined:
        bucket = categories[trade.category]
        for target in (bucket, total):
            target["trade_count"] += 1
            target["base_qty"] += abs(trade.qty)
            target["notional_usd"] += trade.notional()
            target["commission_usd"] += trade.commission
            target["realized_pnl_usd"] += trade.realized_pnl
            if trade.maker is True:
                target["maker_count"] += 1
            elif trade.maker is False:
                target["taker_count"] += 1
            else:
                target["unknown_maker_count"] += 1
        if trade.telemetry_matched:
            matched_count += 1
        if trade.order_matched:
            order_matched_count += 1

    income_sums: dict[str, Decimal] = {}
    for item in income:
        income_type = str(item.get("incomeType") or "UNKNOWN")
        income_sums[income_type] = income_sums.get(income_type, ZERO) + decimal_of(item.get("income"))

    def convert_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
        return {
            "trade_count": bucket["trade_count"],
            "base_qty": decimal_to_float(bucket["base_qty"]),
            "notional_usd": decimal_to_float(bucket["notional_usd"]),
            "commission_usd": decimal_to_float(bucket["commission_usd"]),
            "realized_pnl_usd": decimal_to_float(bucket["realized_pnl_usd"]),
            "maker_count": bucket["maker_count"],
            "taker_count": bucket["taker_count"],
            "unknown_maker_count": bucket["unknown_maker_count"],
        }

    trade_count = len(joined)
    telemetry_match_rate = matched_count / trade_count if trade_count else 0.0
    order_match_rate = order_matched_count / trade_count if trade_count else 0.0
    maker_known_count = total["maker_count"] + total["taker_count"]
    maker_known_rate = maker_known_count / trade_count if trade_count else 0.0

    return {
        "window": {"start_ms": start_ms, "end_ms": end_ms},
        "telemetry": {
            **telemetry_meta,
            "aster_fill_count": telemetry.aster_fill_count,
            "aster_fill_base": decimal_to_float(telemetry.aster_fill_base),
            "aster_mm_fill_count": telemetry.aster_mm_fill_count,
            "aster_hedge_fill_count": telemetry.aster_hedge_fill_count,
            "aster_order_event_count": telemetry.aster_order_event_count,
        },
        "join_quality": {
            "trade_count": trade_count,
            "telemetry_matched_count": matched_count,
            "order_matched_count": order_matched_count,
            "telemetry_match_rate": round(telemetry_match_rate, 6),
            "order_match_rate": round(order_match_rate, 6),
            "maker_known_rate": round(maker_known_rate, 6),
        },
        "totals": convert_bucket(total),
        "categories": {key: convert_bucket(value) for key, value in categories.items()},
        "income_sums": {key: decimal_to_float(value) for key, value in sorted(income_sums.items())},
    }


def recommend_next_child(summary: dict[str, Any]) -> dict[str, Any]:
    join_quality = summary["join_quality"]
    categories = summary["categories"]
    total_commission = abs(decimal_of(summary["totals"]["commission_usd"]))
    trade_count = int(join_quality["trade_count"])
    telemetry_match_rate = float(join_quality["telemetry_match_rate"])
    order_match_rate = float(join_quality["order_match_rate"])
    maker_known_rate = float(join_quality["maker_known_rate"])

    if trade_count == 0:
        return {
            "confidence": "low",
            "recommended_child": "phase5_all5_current_surface_aster_order_truth_join_requal",
            "reason": "Aster venue-native trade query returned zero rows for the promoted run window; attribution cannot validate the fee path.",
        }
    if telemetry_match_rate < 0.75 or order_match_rate < 0.75 or maker_known_rate < 0.75:
        return {
            "confidence": "low",
            "recommended_child": "phase5_all5_current_surface_aster_order_truth_join_requal",
            "reason": "Aster venue-native trades could not be joined to telemetry/order truth at sufficient confidence.",
        }

    confidence = "high" if telemetry_match_rate >= 0.9 and order_match_rate >= 0.9 and maker_known_rate >= 0.9 else "medium"

    def commission(category: str) -> Decimal:
        return abs(decimal_of(categories[category]["commission_usd"]))

    cleanup_commission = commission(CATEGORY_CLEANUP)
    unwind_commission = commission(CATEGORY_TAKER_UNWIND)
    maker_commission = commission(CATEGORY_MAKER_MM)
    unknown_commission = commission(CATEGORY_UNKNOWN) + commission(CATEGORY_UNMATCHED)
    realised = decimal_of(summary["totals"]["realized_pnl_usd"])

    if unknown_commission > total_commission * Decimal("0.10"):
        child = "phase5_all5_current_surface_aster_order_truth_join_requal"
        reason = "Unknown or unmatched Aster fee path remains above 10% of commission; harden order truth before tuning."
    elif cleanup_commission + unwind_commission > total_commission * Decimal("0.50"):
        child = "phase5_all5_current_surface_aster_reduce_only_inventory_brake_fee_guard_requal"
        reason = "Aster fee load is dominated by reduce-only, hedge, unwind, or cleanup paths rather than confirmed maker MM fills."
    elif maker_commission >= total_commission * Decimal("0.50") and realised < ZERO:
        child = "phase5_all5_current_surface_aster_maker_edge_floor_markout_requal"
        reason = "Aster cost is tied to matched MM maker flow with negative realized economics; tune Aster-local edge/markout protection."
    else:
        child = "phase5_reopened_multi_venue_long_soak"
        reason = "Aster fee path is attributable and not dominant; resume all-5 economics clearance once runtime readiness gates pass."

    return {
        "confidence": confidence,
        "recommended_child": child,
        "reason": reason,
        "commission_split_usd": {
            CATEGORY_MAKER_MM: decimal_to_float(maker_commission),
            CATEGORY_TAKER_UNWIND: decimal_to_float(unwind_commission),
            CATEGORY_CLEANUP: decimal_to_float(cleanup_commission),
            CATEGORY_UNKNOWN: decimal_to_float(commission(CATEGORY_UNKNOWN)),
            CATEGORY_UNMATCHED: decimal_to_float(commission(CATEGORY_UNMATCHED)),
        },
    }


def joined_trades_to_csv(path: Path, joined: list[JoinedTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [trade.to_dict() for trade in joined]
    fieldnames = [
        "trade_id",
        "order_id",
        "client_order_id",
        "time_ms",
        "side",
        "qty",
        "price",
        "quote_qty",
        "notional_usd",
        "commission_usd",
        "commission_asset",
        "realized_pnl_usd",
        "fee_bps",
        "maker",
        "time_in_force",
        "reduce_only",
        "post_only",
        "purpose",
        "telemetry_matched",
        "order_matched",
        "category",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_summary_md(
    *,
    collected_at: str,
    run_root: Path,
    summary: dict[str, Any],
    recommendation: dict[str, Any],
    commission_rate: Any,
) -> str:
    lines = [
        "# Phase 5 Aster Fee-Path Attribution Gate",
        "",
        f"- Collected at: `{collected_at}`",
        f"- Run root: `{run_root}`",
        f"- Window: `{summary['window']['start_ms']} -> {summary['window']['end_ms']}`",
        f"- Recommended child: `{recommendation['recommended_child']}`",
        f"- Confidence: `{recommendation['confidence']}`",
        f"- Reason: {recommendation['reason']}",
        "",
        "## Confirmed Internal Facts",
        "",
        f"- Aster telemetry fills: `{summary['telemetry']['aster_fill_count']}` / `{summary['telemetry']['aster_fill_base']} ETH`.",
        f"- Aster MM telemetry fills: `{summary['telemetry']['aster_mm_fill_count']}`.",
        f"- Aster hedge/non-MM telemetry fills: `{summary['telemetry']['aster_hedge_fill_count']}`.",
        f"- Aster order telemetry events: `{summary['telemetry']['aster_order_event_count']}`.",
        "",
        "## Confirmed External Facts",
        "",
        f"- Aster USDT perps charge `0%` maker and `0.04%` taker fees. Source: {ASTER_DOCS['fees']}",
        f"- Aster exposes signed income, trade, order, and commission-rate endpoints used here. Source: {ASTER_DOCS['income']}",
        f"- Aster market-maker incentives require qualification; rebates are not assumed. Source: {ASTER_DOCS['market_maker']}",
        f"- Commission-rate endpoint payload: `{commission_rate}`",
        "",
        "## Join Quality",
        "",
        f"- Venue trade count: `{summary['join_quality']['trade_count']}`",
        f"- Telemetry match rate: `{summary['join_quality']['telemetry_match_rate']}`",
        f"- Order-history match rate: `{summary['join_quality']['order_match_rate']}`",
        f"- Maker/taker known rate: `{summary['join_quality']['maker_known_rate']}`",
        "",
        "## Fee Path Summary",
        "",
        "| Category | Trades | Base | Notional | Commission | Realized PnL | Maker | Taker | Unknown maker |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, payload in summary["categories"].items():
        lines.append(
            "| "
            f"{category} | "
            f"{payload['trade_count']} | "
            f"{payload['base_qty']:.6f} | "
            f"{payload['notional_usd']:.6f} | "
            f"{payload['commission_usd']:.6f} | "
            f"{payload['realized_pnl_usd']:.6f} | "
            f"{payload['maker_count']} | "
            f"{payload['taker_count']} | "
            f"{payload['unknown_maker_count']} |"
        )
    lines.extend(
        [
            "",
            "## Inference",
            "",
            f"- `{recommendation['recommended_child']}` is selected only from the measured fee-path split, not from broad retuning.",
            "- Non-Aster venues remain frozen for this gate.",
            "- If runtime readiness remains false or direct all-5 audit remains non-clean, this attribution result must not launch a live rung by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 Aster fee-path attribution gate")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_ms, end_ms = load_run_window(args.run_root, args.start_ms, args.end_ms)
    env = load_env(args.env_file)
    venue_truth = fetch_aster_venue_truth(env, start_ms=start_ms, end_ms=end_ms)
    telemetry, telemetry_meta = collect_aster_telemetry(args.run_root)
    joined = join_trades(venue_truth["trades"], venue_truth["orders"], telemetry)
    summary = summarize_joined(
        joined,
        venue_truth["income"],
        telemetry,
        telemetry_meta,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    recommendation = recommend_next_child(summary)
    collected_at = utc_now_iso()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "collected_at_utc": collected_at,
        "run_root": str(args.run_root),
        "official_docs": ASTER_DOCS,
        "symbol": venue_truth["symbol"],
        "commission_rate": venue_truth["commission_rate"],
        "summary": summary,
        "recommendation": recommendation,
        "joined_trades": [trade.to_dict() for trade in joined],
    }
    write_json(args.out_dir / "aster_fee_path_ledger.json", payload)
    joined_trades_to_csv(args.out_dir / "aster_fee_path_ledger.csv", joined)
    write_json(args.out_dir / "recommended_child.json", recommendation)
    summary_md = render_summary_md(
        collected_at=collected_at,
        run_root=args.run_root,
        summary=summary,
        recommendation=recommendation,
        commission_rate=venue_truth["commission_rate"],
    )
    write_text(args.out_dir / "aster_fee_path_summary.md", summary_md)
    write_text(args.report_path, summary_md)
    print(summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
