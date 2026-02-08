#!/usr/bin/env python3
"""Terminal dashboard for paraphina_live telemetry.

Displays a colour-coded, Unicode-styled live view of venue status,
positions, fills, cancels, and kill events.  All rendering is pure
display logic – no market-making behaviour is changed.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable

# ── ANSI styling ──────────────────────────────────────────────────────────────

_NO_COLOR = False  # flipped by --no-color / NO_COLOR env / non-TTY

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class S:
    """ANSI escape sequences for styles and colours."""

    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    # foreground
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"
    GRAY = "\x1b[90m"
    # bright foreground
    B_RED = "\x1b[91m"
    B_GREEN = "\x1b[92m"
    B_YELLOW = "\x1b[93m"
    B_CYAN = "\x1b[96m"
    B_WHITE = "\x1b[97m"


def _s(*codes: str) -> str:
    """Join style codes (empty when colour is off)."""
    return "" if _NO_COLOR else "".join(codes)


def _r() -> str:
    """Reset code (empty when colour is off)."""
    return "" if _NO_COLOR else S.RESET


def visible_len(text: str) -> int:
    """Visible width of *text*, ignoring ANSI escapes."""
    return len(_ANSI_RE.sub("", text))


def styled(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI codes with auto-reset."""
    if _NO_COLOR or not codes:
        return text
    return "".join(codes) + text + S.RESET


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal dashboard for paraphina_live telemetry."
    )
    parser.add_argument("--telemetry", required=True, help="Path to telemetry.jsonl")
    parser.add_argument("--refresh-ms", type=int, default=250)
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument(
        "--no-color", action="store_true", help="Disable coloured output"
    )
    return parser.parse_args()


# ── Value helpers (unchanged logic) ───────────────────────────────────────────


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


# ── Colour helpers ────────────────────────────────────────────────────────────

_LABEL = lambda t: styled(t, S.GRAY)  # noqa: E731  dim label


def color_health(status: str) -> str:
    if status == "Healthy":
        return styled(status, S.GREEN)
    if status in ("Stale", "Disconnected", "Error"):
        return styled(status, S.B_RED, S.BOLD)
    return styled(status, S.YELLOW)


def color_regime(regime: str) -> str:
    if regime == "Normal":
        return styled(regime, S.GREEN)
    if regime in ("Emergency", "HardStop"):
        return styled(regime, S.B_RED, S.BOLD)
    return styled(regime, S.YELLOW)


def color_kill(kill: bool) -> str:
    if kill:
        return styled("True", S.B_RED, S.BOLD)
    return styled("False", S.GREEN)


def _color_val(text: str, value: float | None, lo: float, hi: float) -> str:
    """Colour a pre-formatted string based on absolute-value thresholds."""
    if value is None:
        return styled(text, S.GRAY)
    v = abs(value)
    if v >= hi:
        return styled(text, S.B_RED)
    if v >= lo:
        return styled(text, S.YELLOW)
    return text


def color_tox(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return styled("n/a", S.GRAY)
    text = f"{value:.{decimals}f}"
    return _color_val(text, value, 0.2, 0.5)


def color_stale(pct: float) -> str:
    text = f"{pct:.1f}%"
    if pct >= 5.0:
        return styled(text, S.B_RED)
    if pct >= 1.0:
        return styled(text, S.YELLOW)
    return styled(text, S.GREEN)


# ── Venue-ID parsing (unchanged logic) ───────────────────────────────────────


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
                return [
                    mapping.get(i, f"venue_{i}")
                    for i in range(max(mapping.keys()) + 1)
                ]
    return [f"venue_{i}" for i in range(fallback_count)]


# ── Telemetry parsing (unchanged logic) ──────────────────────────────────────


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


# ── State tracking (unchanged logic) ─────────────────────────────────────────


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
            cur = (
                venue_status[idx]
                if isinstance(venue_status, list) and idx < len(venue_status)
                else None
            )
            if isinstance(cur, str):
                if cur != "Healthy":
                    self.venue_stale_ticks[vid] = (
                        self.venue_stale_ticks.get(vid, 0) + 1
                    )
                prev = self.prev_venue_status.get(vid)
                if prev is not None and cur != prev:
                    self.venue_status_flips[vid] = (
                        self.venue_status_flips.get(vid, 0) + 1
                    )
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
                    age = (
                        f"{int((now_ms - fill_time) / 1000)}s"
                        if now_ms and fill_time
                        else "n/a"
                    )
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


# ── Table formatting (Unicode box-drawing) ───────────────────────────────────

_DISPLAY_EVENT_LIMIT = 10  # max events shown per section in the dashboard


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build an aligned table with Unicode separators.

    Cell values may contain ANSI codes; ``visible_len`` is used for
    width calculations so alignment is correct.
    """
    col_count = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for idx in range(min(len(row), col_count)):
            widths[idx] = max(widths[idx], visible_len(row[idx]))

    dim = _s(S.DIM)
    rst = _r()
    col_sep = f" {dim}│{rst} "

    # Header row
    header_cells = [
        styled(h.ljust(widths[i]), S.BOLD, S.CYAN) for i, h in enumerate(headers)
    ]
    header_line = col_sep.join(header_cells)

    # Separator row
    sep_parts = ["─" * widths[i] for i in range(col_count)]
    sep_line = f"{dim}{'─┼─'.join(sep_parts)}{rst}"

    # Data rows
    lines = [header_line, sep_line]
    for row in rows:
        cells = []
        for i in range(col_count):
            cell = row[i] if i < len(row) else ""
            pad = widths[i] - visible_len(cell)
            cells.append(cell + " " * pad)
        lines.append(col_sep.join(cells))

    return "\n".join(lines)


# ── Section header ────────────────────────────────────────────────────────────


def _section(title: str, width: int = 72) -> str:
    """Render ``─── Title ────────────────``."""
    prefix = "─── "
    suffix_len = max(1, width - len(prefix) - len(title) - 1)
    return styled(f"{prefix}{title} {'─' * suffix_len}", S.CYAN, S.BOLD)


# ── Frame rendering ──────────────────────────────────────────────────────────


def render_frame(state: WatchState, max_events: int) -> str:  # noqa: C901
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

    # ── Title bar ─────────────────────────────────────────────────────────
    rule = styled("━" * 72, S.CYAN)
    title = styled("  PARAPHINA LIVE WATCH", S.B_CYAN, S.BOLD)

    # ── Header metrics ────────────────────────────────────────────────────
    tick_str = styled(str(tick), S.B_WHITE) if tick is not None else styled("n/a", S.GRAY)
    time_str = (
        styled(short_ts(now_ms), S.B_WHITE, S.BOLD) if now_ms else styled("n/a", S.GRAY)
    )
    mode_str = styled(str(execution_mode), S.MAGENTA)
    trade_str = styled(str(trade_mode), S.MAGENTA)
    regime_str = color_regime(str(risk_regime))
    kill_str = color_kill(bool(kill_switch))

    hdr1 = (
        f"  {_LABEL('tick')} {tick_str}   "
        f"{_LABEL('time')} {time_str}   "
        f"{_LABEL('mode')} {mode_str}   "
        f"{_LABEL('trade')} {trade_str}"
    )
    hdr2 = f"  {_LABEL('regime')} {regime_str}   {_LABEL('kill')} {kill_str}"
    if kill_switch and kill_reason and kill_reason != "n/a":
        hdr2 += f"   {_LABEL('reason')} {styled(str(kill_reason), S.B_RED)}"

    # Position / PnL
    q_str = (
        _color_val(f"{float(q_global):.4f}", safe_float(q_global), 0.1, 1.0)
        if q_global is not None
        else styled("0.0", S.DIM)
    )
    delta_str = str(delta_usd) if delta_usd is not None else "0.0"
    basis_str = str(basis) if basis is not None else "0.0"
    basis_g_str = str(basis_gross) if basis_gross is not None else "0.0"

    hdr3 = (
        f"  {_LABEL('q_global')} {q_str}   "
        f"{_LABEL('Δ_usd')} {delta_str}   "
        f"{_LABEL('basis')} {basis_str}   "
        f"{_LABEL('basis_gross')} {basis_g_str}"
    )

    # Toxicity
    hdr4 = (
        f"  {_LABEL('tox_avg')} {color_tox(tox_avg)}   "
        f"{_LABEL('tox_max')} {color_tox(tox_max)}"
    )

    lines: list[str] = [rule, title, rule, hdr1, hdr2, hdr3, hdr4, ""]

    # ── Venue table ───────────────────────────────────────────────────────
    venue_ids = state.venue_ids
    v_status = record.get("venue_status", [])
    v_mid = record.get("venue_mid_usd", [])
    v_spread = record.get("venue_spread_usd", [])
    v_age = record.get("venue_age_ms", [])
    v_pos = record.get("venue_position_tao", [])
    v_fund_rate = record.get("venue_funding_rate_8h", [])
    v_fund_age = record.get("venue_funding_age_ms", [])
    v_fund_status = record.get("venue_funding_status", [])
    orders_raw = record.get("orders", [])
    fills_raw = record.get("fills", [])
    if not isinstance(orders_raw, list):
        orders_raw = []
    if not isinstance(fills_raw, list):
        fills_raw = []

    order_counts: dict[str, int] = {}
    for order in orders_raw:
        if not isinstance(order, dict):
            continue
        vid = order.get("venue_id")
        if isinstance(vid, str):
            order_counts[vid] = order_counts.get(vid, 0) + 1

    fill_counts: dict[str, int] = {}
    for fill_item in fills_raw:
        if not isinstance(fill_item, dict):
            continue
        vid = fill_item.get("venue_id")
        if isinstance(vid, str):
            fill_counts[vid] = fill_counts.get(vid, 0) + 1

    rows: list[list[str]] = []
    for idx, venue_id in enumerate(venue_ids):
        status_val = v_status[idx] if idx < len(v_status) else None
        mid_val = v_mid[idx] if idx < len(v_mid) else None
        spread_val = v_spread[idx] if idx < len(v_spread) else None
        age_val = v_age[idx] if idx < len(v_age) else None
        pos_val = v_pos[idx] if idx < len(v_pos) else None
        fund_rate_val = v_fund_rate[idx] if idx < len(v_fund_rate) else None
        fund_age_val = v_fund_age[idx] if idx < len(v_fund_age) else None
        fund_status_val = v_fund_status[idx] if idx < len(v_fund_status) else None
        open_orders = order_counts.get(venue_id, 0)
        last_fill_ms = state.last_fill_ms.get(venue_id)
        last_fill_age = styled("n/a", S.GRAY)
        if now_ms is not None and last_fill_ms is not None:
            last_fill_age = f"{int((now_ms - last_fill_ms) / 1000)}s"

        # Health + toxicity cell
        health_s = color_health(format_status(status_val))
        tox_val = tox[idx] if isinstance(tox, list) and idx < len(tox) else None
        tox_s = color_tox(
            tox_val if isinstance(tox_val, (int, float)) else None, decimals=2
        )

        # Stale% / flips cell
        stale_ticks = state.venue_stale_ticks.get(venue_id, 0)
        stale_pct = (
            (100.0 * stale_ticks / state.tick_count) if state.tick_count > 0 else 0.0
        )
        flips = state.venue_status_flips.get(venue_id, 0)
        stale_s = color_stale(stale_pct)

        # Formatted cells (numbers first, then colour)
        mid_f = format_num(mid_val, 10).strip()
        spread_f = format_num(spread_val, 8).strip()
        spread_f = _color_val(spread_f, safe_float(spread_val), 0.5, 2.0)
        age_f = format_ms(age_val)
        age_f = _color_val(age_f, safe_float(safe_int(age_val)), 2000, 5000)
        pos_f = format_num(pos_val, 8).strip()
        pos_f = _color_val(pos_f, safe_float(pos_val), 0.1, 1.0)
        fund_rate_f = format_num(fund_rate_val, 8).strip()
        fund_age_f = format_ms(fund_age_val)
        fund_status_f = color_health(format_status(fund_status_val))
        orders_f = str(open_orders)

        rows.append(
            [
                styled(venue_id, S.BOLD, S.WHITE),
                mid_f,
                spread_f,
                age_f,
                pos_f,
                fund_rate_f,
                fund_age_f,
                fund_status_f,
                orders_f,
                last_fill_age,
                f"{health_s} {_LABEL('tox=')}{tox_s}",
                f"{stale_s}{_LABEL('/')}{str(flips)}",
            ]
        )

    lines.append(_section("Venues"))
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

    # ── Event logs ────────────────────────────────────────────────────────
    def _event_section(
        title: str,
        events: Deque[str],
        bullet_color: str,
    ) -> None:
        count = len(events)
        lines.append("")
        lines.append(_section(f"{title} ({count})"))
        shown = list(events)[:_DISPLAY_EVENT_LIMIT]
        if shown:
            for item in shown:
                lines.append(f"  {styled('●', bullet_color)} {item}")
            remaining = count - len(shown)
            if remaining > 0:
                lines.append(
                    f"  {styled(f'… and {remaining} more', S.GRAY)}"
                )
        else:
            lines.append(f"  {styled('(none)', S.GRAY)}")

    _event_section("Recent Fills", state.events.fills, S.GREEN)
    _event_section("Recent Cancels", state.events.cancels, S.YELLOW)
    _event_section("Recent Kills", state.events.kills, S.B_RED)

    return "\n".join(lines)


# ── Tail follower (unchanged logic) ──────────────────────────────────────────


class TailFollower:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def seek_end(self) -> None:
        """Advance offset to end of file so subsequent reads only see new data."""
        try:
            self.offset = self.path.stat().st_size
        except OSError:
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


# ── Entry points ─────────────────────────────────────────────────────────────


def render_once(path: Path, refresh_ms: int, max_events: int) -> str:
    records = parse_lines(path, max_events)
    state = build_state(records, max_events)
    return render_frame(state, max_events)


def main() -> int:
    global _NO_COLOR
    args = parse_args()
    telemetry_path = Path(args.telemetry)
    max_events = max(1, args.max_events)
    refresh_ms = max(1, args.refresh_ms)

    if args.no_color or os.environ.get("NO_COLOR"):
        _NO_COLOR = True

    one_shot = refresh_ms >= 999_999
    if one_shot:
        frame = render_once(telemetry_path, refresh_ms, max_events)
        print(frame)
        return 0

    is_tty = sys.stdout.isatty()
    if not is_tty:
        _NO_COLOR = True

    initial_records = parse_lines(telemetry_path, max_events)
    state = build_state(initial_records, max_events)
    follower = TailFollower(telemetry_path)
    # Advance follower past already-consumed data so we don't double-count.
    follower.seek_end()

    if is_tty:
        # Hide cursor and clear screen once at startup.
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
        sys.stdout.flush()

    def cleanup() -> None:
        if is_tty:
            sys.stdout.write("\x1b[?25h")  # Show cursor again.
            sys.stdout.flush()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))

    while True:
        for line in follower.read_new_lines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                state.update(record)
        frame = render_frame(state, max_events)
        if is_tty:
            # Move cursor home, then write each line with a clear-to-EOL
            # escape so stale characters from longer previous lines are
            # erased.  Finally clear everything below the frame.
            sys.stdout.write("\x1b[H")
            for fline in frame.split("\n"):
                sys.stdout.write(fline + "\x1b[K\n")
            sys.stdout.write("\x1b[J")
        else:
            sys.stdout.write(frame + "\n")
        sys.stdout.flush()
        time.sleep(refresh_ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
