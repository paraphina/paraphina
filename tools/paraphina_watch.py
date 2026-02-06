#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal dashboard for paraphina_live telemetry.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry.jsonl")
    parser.add_argument("--refresh-ms", type=int, default=250)
    parser.add_argument("--max-events", type=int, default=50)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def safe_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def short_ts(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%H:%M:%S")


def format_num(value: Any, width: int = 8) -> str:
    if value is None:
        return " " * (width - 3) + "n/a"
    if isinstance(value, float):
        return f"{value:>{width}.4f}"
    return f"{str(value):>{width}}"


def format_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{int(value)}"
    return "n/a"


def format_status(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "n/a"


def parse_venue_ids(record: dict[str, Any], fallback_count: int) -> list[str]:
    treasury = record.get("treasury_guidance")
    if isinstance(treasury, dict):
        venues = treasury.get("venues")
        if isinstance(venues, list):
            mapping = {}
            for venue in venues:
                if not isinstance(venue, dict):
                    continue
                idx = safe_int(venue.get("venue_index"))
                name = venue.get("venue_id")
                if idx is not None and isinstance(name, str):
                    mapping[idx] = name
            if mapping:
                return [mapping.get(i, f"venue_{i}") for i in range(max(mapping.keys()) + 1)]
    return [f"venue_{i}" for i in range(fallback_count)]


def parse_lines(path: Path, max_events: int) -> list[dict[str, Any]]:
    records: Deque[dict[str, Any]] = deque(maxlen=max_events)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        return []
    return list(records)


@dataclass
class EventLog:
    fills: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    cancels: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    kills: Deque[str] = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class WatchState:
    last_record: dict[str, Any] | None = None
    venue_ids: list[str] = field(default_factory=list)
    last_fill_ms: dict[str, int] = field(default_factory=dict)
    events: EventLog = field(default_factory=EventLog)
    tick_count: int = 0
    prev_venue_status: dict[str, str] = field(default_factory=dict)
    venue_status_flips: dict[str, int] = field(default_factory=dict)
    venue_stale_ticks: dict[str, int] = field(default_factory=dict)

    def update(self, record: dict[str, Any]) -> None:
        self.last_record = record
        self.tick_count += 1
        venue_status = record.get("venue_status", [])
        venue_count = len(venue_status) if isinstance(venue_status, list) else 0
        self.venue_ids = parse_venue_ids(record, venue_count)

        # Track per-venue stale% and status flips.
        for idx, vid in enumerate(self.venue_ids):
            cur = venue_status[idx] if isinstance(venue_status, list) and idx < len(venue_status) else None
            if isinstance(cur, str):
                if cur != "Healthy":
                    self.venue_stale_ticks[vid] = self.venue_stale_ticks.get(vid, 0) + 1
                prev = self.prev_venue_status.get(vid)
                if prev is not None and cur != prev:
                    self.venue_status_flips[vid] = self.venue_status_flips.get(vid, 0) + 1
                self.prev_venue_status[vid] = cur

        now_ms = None
        treasury = record.get("treasury_guidance")
        if isinstance(treasury, dict):
            now_ms = safe_int(treasury.get("as_of_ms"))

        fills = record.get("fills", [])
        if isinstance(fills, list):
            for fill in fills:
                if not isinstance(fill, dict):
                    continue
                venue_id = fill.get("venue_id")
                if isinstance(venue_id, str):
                    fill_time = safe_int(fill.get("fill_time_ms"))
                    if fill_time is not None:
                        self.last_fill_ms[venue_id] = fill_time
                    size = fill.get("size")
                    price = fill.get("price")
                    side = fill.get("side", "?")
                    age = f"{int((now_ms - fill_time) / 1000)}s" if now_ms and fill_time else "n/a"
                    self.events.fills.appendleft(
                        f"{venue_id} {side} {size}@{price} age={age}"
                    )

        orders = record.get("orders", [])
        if isinstance(orders, list):
            for order in orders:
                if not isinstance(order, dict):
                    continue
                if order.get("action") == "cancel" and order.get("status") == "ack":
                    venue_id = order.get("venue_id", "n/a")
                    reason = order.get("reason", "")
                    suffix = f" reason={reason}" if reason else ""
                    self.events.cancels.appendleft(f"{venue_id} cancel{suffix}")

        if record.get("kill_switch"):
            reason = record.get("kill_reason", "unknown")
            tick = record.get("t", "n/a")
            self.events.kills.appendleft(f"tick={tick} reason={reason}")


def build_state(records: Iterable[dict[str, Any]], max_events: int) -> WatchState:
    state = WatchState()
    state.events = EventLog(
        fills=deque(maxlen=max_events),
        cancels=deque(maxlen=max_events),
        kills=deque(maxlen=max_events),
    )
    for record in records:
        state.update(record)
    return state


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = []
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def render_frame(state: WatchState, max_events: int) -> str:
    record = state.last_record or {}
    tick = record.get("t")
    now_ms = None
    treasury = record.get("treasury_guidance")
    if isinstance(treasury, dict):
        now_ms = safe_int(treasury.get("as_of_ms"))
    execution_mode = record.get("execution_mode", "n/a")
    trade_mode = record.get("trade_mode", execution_mode)
    risk_regime = record.get("risk_regime", "n/a")
    kill_switch = record.get("kill_switch", False)
    kill_reason = record.get("kill_reason", "n/a")
    q_global = record.get("q_global_tao")
    delta_usd = record.get("dollar_delta_usd")
    basis = record.get("basis_usd")
    basis_gross = record.get("basis_gross_usd")

    tox = record.get("venue_toxicity", [])
    tox_values = [v for v in tox if isinstance(v, (int, float))]
    tox_avg = sum(tox_values) / len(tox_values) if tox_values else None
    tox_max = max(tox_values) if tox_values else None

    lines = [
        "Paraphina Live Watch",
        f"tick={tick} time={short_ts(now_ms)} mode={execution_mode} trade={trade_mode}",
        f"regime={risk_regime} kill={kill_switch} reason={kill_reason}",
        f"q_global={q_global} delta_usd={delta_usd} basis={basis} basis_gross={basis_gross}",
        f"tox_avg={tox_avg:.4f}" if tox_avg is not None else "tox_avg=n/a",
        f"tox_max={tox_max:.4f}" if tox_max is not None else "tox_max=n/a",
        "",
    ]

    venue_ids = state.venue_ids
    status = record.get("venue_status", [])
    mid = record.get("venue_mid_usd", [])
    spread = record.get("venue_spread_usd", [])
    age = record.get("venue_age_ms", [])
    pos = record.get("venue_position_tao", [])
    funding_rate = record.get("venue_funding_rate_8h", [])
    funding_age = record.get("venue_funding_age_ms", [])
    funding_status = record.get("venue_funding_status", [])
    orders = record.get("orders", [])
    fills = record.get("fills", [])
    if not isinstance(orders, list):
        orders = []
    if not isinstance(fills, list):
        fills = []
    order_counts = {}
    for order in orders:
        if not isinstance(order, dict):
            continue
        venue_id = order.get("venue_id")
        if isinstance(venue_id, str):
            order_counts[venue_id] = order_counts.get(venue_id, 0) + 1

    fill_counts = {}
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        venue_id = fill.get("venue_id")
        if isinstance(venue_id, str):
            fill_counts[venue_id] = fill_counts.get(venue_id, 0) + 1

    rows = []
    for idx, venue_id in enumerate(venue_ids):
        status_val = status[idx] if idx < len(status) else None
        mid_val = mid[idx] if idx < len(mid) else None
        spread_val = spread[idx] if idx < len(spread) else None
        age_val = age[idx] if idx < len(age) else None
        pos_val = pos[idx] if idx < len(pos) else None
        funding_rate_val = funding_rate[idx] if idx < len(funding_rate) else None
        funding_age_val = funding_age[idx] if idx < len(funding_age) else None
        funding_status_val = funding_status[idx] if idx < len(funding_status) else None
        open_orders = order_counts.get(venue_id, 0)
        last_fill_ms = state.last_fill_ms.get(venue_id)
        last_fill_age = "n/a"
        if now_ms is not None and last_fill_ms is not None:
            last_fill_age = f"{int((now_ms - last_fill_ms) / 1000)}s"
        health = format_status(status_val)
        tox_val = tox[idx] if isinstance(tox, list) and idx < len(tox) else None
        tox_str = f"{tox_val:.2f}" if isinstance(tox_val, (int, float)) else "n/a"
        # Compute cumulative stale% and flip count for this venue.
        stale_ticks = state.venue_stale_ticks.get(venue_id, 0)
        stale_pct = (100.0 * stale_ticks / state.tick_count) if state.tick_count > 0 else 0.0
        flips = state.venue_status_flips.get(venue_id, 0)
        stale_flips_str = f"{stale_pct:.1f}%/{flips}"
        rows.append(
            [
                venue_id,
                format_num(mid_val, 10).strip(),
                format_num(spread_val, 8).strip(),
                format_ms(age_val),
                format_num(pos_val, 8).strip(),
                format_num(funding_rate_val, 8).strip(),
                format_ms(funding_age_val),
                format_status(funding_status_val),
                str(open_orders),
                last_fill_age,
                f"{health} tox={tox_str}",
                stale_flips_str,
            ]
        )

    lines.append(
        format_table(
            [
                "venue",
                "mid",
                "spread",
                "age_ms",
                "pos",
                "fund_8h",
                "fund_age",
                "fund_status",
                "orders",
                "last_fill",
                "health",
                "stale%/flips",
            ],
            rows,
        )
    )

    lines.append("")
    lines.append("recent fills:")
    for item in list(state.events.fills)[:max_events]:
        lines.append(f"  {item}")
    if not state.events.fills:
        lines.append("  (none)")
    lines.append("")
    lines.append("recent cancels:")
    for item in list(state.events.cancels)[:max_events]:
        lines.append(f"  {item}")
    if not state.events.cancels:
        lines.append("  (none)")
    lines.append("")
    lines.append("recent kill events:")
    for item in list(state.events.kills)[:max_events]:
        lines.append(f"  {item}")
    if not state.events.kills:
        lines.append("  (none)")
    return "\n".join(lines)


class TailFollower:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_new_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            self.offset = 0
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self.offset)
                data = handle.read()
                self.offset = handle.tell()
        except OSError:
            return []
        if not data:
            return []
        return [line for line in data.splitlines() if line.strip()]


def render_once(path: Path, refresh_ms: int, max_events: int) -> str:
    records = parse_lines(path, max_events)
    state = build_state(records, max_events)
    return render_frame(state, max_events)


def main() -> int:
    args = parse_args()
    telemetry_path = Path(args.telemetry)
    max_events = max(1, args.max_events)
    refresh_ms = max(1, args.refresh_ms)

    one_shot = refresh_ms >= 999_999
    if one_shot:
        frame = render_once(telemetry_path, refresh_ms, max_events)
        print(frame)
        return 0

    initial_records = parse_lines(telemetry_path, max_events)
    state = build_state(initial_records, max_events)
    follower = TailFollower(telemetry_path)

    while True:
        for line in follower.read_new_lines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                state.update(record)
        frame = render_frame(state, max_events)
        if sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(frame + "\n")
        sys.stdout.flush()
        time.sleep(refresh_ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
