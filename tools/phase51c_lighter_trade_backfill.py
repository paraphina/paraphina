#!/usr/bin/env python3
"""Collect paginated Lighter native trades for Phase 5.1c attribution.

This is a read-only evidence utility. It never submits orders, never changes
capital/risk state, and only emits sanitized artifacts for maker/taker role
attribution review.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase51b_lighter_account_limits import (
    DEFAULT_BASE_URL,
    READONLY_ENDPOINTS,
    _artifact_infos,
    _as_int,
    _extract_items,
    _get_auth_token,
    _http_get_json,
    _load_env_file,
    _redact,
    _resolve_base_url,
    _sha256_file,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_lighter_trade_backfill"
OFFICIAL_TRADES_DOC_URL = "https://apidocs.lighter.xyz/reference/trades"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _trade_timestamp_ms(trade: dict[str, Any]) -> int | None:
    return _as_int(trade.get("timestamp") or trade.get("time") or trade.get("created_at"))


def _role_counts(trades: list[dict[str, Any]], account_index: int | None) -> dict[str, int]:
    counts = {"maker": 0, "taker": 0, "unknown": 0}
    for trade in trades:
        is_maker_ask = trade.get("is_maker_ask")
        if is_maker_ask is None:
            is_maker_ask = trade.get("isMakerAsk")
        if not isinstance(is_maker_ask, bool) or account_index is None:
            counts["unknown"] += 1
            continue
        ask_account_id = _as_int(trade.get("ask_account_id") or trade.get("askAccountId"))
        bid_account_id = _as_int(trade.get("bid_account_id") or trade.get("bidAccountId"))
        if account_index == ask_account_id:
            counts["maker" if is_maker_ask else "taker"] += 1
        elif account_index == bid_account_id:
            counts["taker" if is_maker_ask else "maker"] += 1
        else:
            counts["unknown"] += 1
    return counts


def _summarize_trades(trades: list[dict[str, Any]], account_index: int | None) -> dict[str, Any]:
    timestamps = [ts for ts in (_trade_timestamp_ms(trade) for trade in trades) if ts is not None]
    return {
        "trade_count": len(trades),
        "timestamp_min_ms": min(timestamps) if timestamps else None,
        "timestamp_max_ms": max(timestamps) if timestamps else None,
        "role_counts_for_account": _role_counts(trades, account_index),
    }


def _page_next_cursor(payload: Any) -> str | None:
    if isinstance(payload, dict):
        cursor = payload.get("next_cursor") or payload.get("nextCursor")
        if cursor:
            return str(cursor)
    return None


def _page_trades(payload: Any) -> list[dict[str, Any]]:
    return _extract_items(payload, {"trades", "trade_history", "tradehistory"})


def _fetch_pages(
    *,
    env: dict[str, str],
    base_url: str,
    account_index: int,
    market_id: int | None,
    market_type: str,
    allow_sdk_auth: bool,
    sdk_path: Path | None,
    timeout_s: float,
    pages: int,
    limit: int,
    start_cursor: str | None,
    from_timestamp_ms: int | None,
    stop_at_or_before_ms: int | None,
    sleep_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    auth_token = _get_auth_token(env, allow_sdk_auth=allow_sdk_auth, sdk_path=sdk_path)
    endpoint = READONLY_ENDPOINTS["trades"]
    fetched_pages: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    cursor = start_cursor
    last_next_cursor: str | None = None
    for page_index in range(pages):
        params = {
            "account_index": account_index,
            "market_id": market_id,
            "market_type": market_type,
            "sort_by": "timestamp",
            "sort_dir": "desc",
            "limit": limit,
            "cursor": cursor,
            "from": from_timestamp_ms if cursor is None else None,
        }
        payload = _http_get_json(
            base_url,
            endpoint,
            params=params,
            auth_token=auth_token,
            timeout_s=timeout_s,
        )
        trades = _page_trades(payload)
        fetched_pages.append({
            "page_index": page_index,
            "params": {k: v for k, v in params.items() if v is not None},
            "payload": payload,
            "trade_count": len(trades),
            "timestamp_min_ms": min(
                (ts for ts in (_trade_timestamp_ms(trade) for trade in trades) if ts is not None),
                default=None,
            ),
            "timestamp_max_ms": max(
                (ts for ts in (_trade_timestamp_ms(trade) for trade in trades) if ts is not None),
                default=None,
            ),
        })
        all_trades.extend(trades)
        last_next_cursor = _page_next_cursor(payload)
        page_min_ts = fetched_pages[-1]["timestamp_min_ms"]
        if stop_at_or_before_ms is not None and page_min_ts is not None and page_min_ts <= stop_at_or_before_ms:
            break
        if not trades or not last_next_cursor:
            break
        cursor = last_next_cursor
        if page_index + 1 < pages and sleep_s > 0:
            time.sleep(sleep_s)
    return fetched_pages, all_trades, last_next_cursor


def _load_page_files(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    pages: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    last_cursor: str | None = None
    for page_index, path in enumerate(paths):
        payload = _load_json(path)
        page_trades = _page_trades(payload)
        last_cursor = _page_next_cursor(payload)
        pages.append({
            "page_index": page_index,
            "source_path": str(path),
            "payload": payload,
            "trade_count": len(page_trades),
            "timestamp_min_ms": min(
                (ts for ts in (_trade_timestamp_ms(trade) for trade in page_trades) if ts is not None),
                default=None,
            ),
            "timestamp_max_ms": max(
                (ts for ts in (_trade_timestamp_ms(trade) for trade in page_trades) if ts is not None),
                default=None,
            ),
        })
        trades.extend(page_trades)
    return pages, trades, last_cursor


def build_trade_backfill(
    *,
    env_file: Path | None,
    page_json: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    account_index: int | None,
    market_id: int | None,
    market_type: str,
    base_url: str | None,
    allow_sdk_auth: bool,
    sdk_path: Path | None,
    timeout_s: float,
    pages: int,
    limit: int,
    start_cursor: str | None,
    from_timestamp_ms: int | None,
    stop_at_or_before_ms: int | None,
    sleep_s: float,
) -> Path:
    run_id = run_id or f"PHASE51C-LIGHTER-TRADE-BACKFILL-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    source_dir = out_dir / "source_snapshots"
    source_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    if page_json:
        fetched_pages, trades, next_cursor = _load_page_files(page_json)
        resolved_account_index = account_index
        resolved_base_url = base_url or DEFAULT_BASE_URL
        source_mode = "offline_page_json"
    else:
        if env_file is None:
            raise ValueError("--env-file is required unless --page-json is supplied")
        env = _load_env_file(env_file)
        resolved_account_index = account_index if account_index is not None else int(env["LIGHTER_ACCOUNT_INDEX"])
        resolved_base_url = _resolve_base_url(env, base_url)
        fetched_pages, trades, next_cursor = _fetch_pages(
            env=env,
            base_url=resolved_base_url,
            account_index=resolved_account_index,
            market_id=market_id,
            market_type=market_type,
            allow_sdk_auth=allow_sdk_auth,
            sdk_path=sdk_path,
            timeout_s=timeout_s,
            pages=pages,
            limit=limit,
            start_cursor=start_cursor,
            from_timestamp_ms=from_timestamp_ms,
            stop_at_or_before_ms=stop_at_or_before_ms,
            sleep_s=sleep_s,
        )
        source_mode = "readonly_lighter_api"

    pages_path = source_dir / "trades_backfill_pages.sanitized.json"
    trades_path = source_dir / "trades_backfill.sanitized.json"
    _write_json(pages_path, {
        "schema_version": 1,
        "source_mode": source_mode,
        "official_docs": [OFFICIAL_TRADES_DOC_URL],
        "pages": _redact(fetched_pages),
    })
    trade_summary = _summarize_trades(trades, resolved_account_index)
    _write_json(trades_path, {
        "schema_version": 1,
        "source_mode": source_mode,
        "official_docs": [OFFICIAL_TRADES_DOC_URL],
        "account_index": resolved_account_index,
        "market_id": market_id,
        "market_type": market_type,
        "trades": _redact(trades),
        **trade_summary,
    })

    complete_to_requested_stop = False
    if stop_at_or_before_ms is not None:
        min_ts = trade_summary["timestamp_min_ms"]
        complete_to_requested_stop = min_ts is not None and min_ts <= stop_at_or_before_ms
    summary = {
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "native_trade_backfill_readonly_attribution_input_only",
        "source_mode": source_mode,
        "official_docs": [OFFICIAL_TRADES_DOC_URL],
        "base_url": resolved_base_url,
        "account_index_present": resolved_account_index is not None,
        "market_id": market_id,
        "market_type": market_type,
        "pages_requested": pages,
        "pages_fetched": len(fetched_pages),
        "limit": limit,
        "start_cursor_present": bool(start_cursor),
        "from_timestamp_ms": from_timestamp_ms,
        "stop_at_or_before_ms": stop_at_or_before_ms,
        "complete_to_requested_stop": complete_to_requested_stop,
        "next_cursor_present": bool(next_cursor),
        "next_cursor": next_cursor,
        "trades_path": str(trades_path),
        "trades_sha256": _sha256_file(trades_path),
        "pages_sha256": _sha256_file(pages_path),
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_financial_claim": False,
        "admissible_for_financial_claim": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        **trade_summary,
    }
    summary_path = out_dir / "lighter_trade_backfill_summary.json"
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [pages_path, trades_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [pages_path, trades_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--page-json", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--account-index", type=int, default=None)
    parser.add_argument("--market-id", type=int, default=0)
    parser.add_argument("--market-type", default="perp", choices=["all", "spot", "perp"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--allow-sdk-auth", action="store_true")
    parser.add_argument("--lighter-sdk-path", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-cursor", default=None)
    parser.add_argument("--from-timestamp-ms", type=int, default=None)
    parser.add_argument("--stop-at-or-before-ms", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=1.6)
    args = parser.parse_args()
    if args.pages < 1:
        print("phase51c_lighter_trade_backfill: ERROR: --pages must be >= 1", file=sys.stderr)
        return 2
    if args.limit < 1 or args.limit > 100:
        print("phase51c_lighter_trade_backfill: ERROR: --limit must be between 1 and 100", file=sys.stderr)
        return 2
    try:
        out_dir = build_trade_backfill(
            env_file=args.env_file,
            page_json=args.page_json,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            account_index=args.account_index,
            market_id=args.market_id,
            market_type=args.market_type,
            base_url=args.base_url,
            allow_sdk_auth=args.allow_sdk_auth,
            sdk_path=args.lighter_sdk_path,
            timeout_s=args.timeout_s,
            pages=args.pages,
            limit=args.limit,
            start_cursor=args.start_cursor,
            from_timestamp_ms=args.from_timestamp_ms,
            stop_at_or_before_ms=args.stop_at_or_before_ms,
            sleep_s=args.sleep_s,
        )
    except Exception as exc:
        print(f"phase51c_lighter_trade_backfill: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_lighter_trade_backfill: wrote {out_dir}")
    print("phase51c_lighter_trade_backfill: status HOLD (read-only attribution input only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
