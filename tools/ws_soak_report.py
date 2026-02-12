#!/usr/bin/env python3
"""
ws_soak_report.py - Build a markdown evidence pack from a shadow soak out-dir.

Expected files under --out-dir:
  - telemetry.jsonl (required)
  - run.log (required)
  - market_rx_stats.log (optional)

Usage:
  python3 tools/ws_soak_report.py --out-dir /tmp/ws_soak
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLATEAU_THRESHOLDS_MS = (10_000, 30_000)
PLATEAU_MIN_TICKS = 3
VENUE_HINTS = ("extended", "hyperliquid", "aster", "lighter", "paradex")
APPLY_P95_MAX_MS = 10_000.0
APPLY_P99_MAX_MS = 30_000.0
EVENT_P95_MAX_MS = 12_000.0
EVENT_P99_MAX_MS = 35_000.0
PLATEAU_GATE_THRESHOLD_MS = 30_000
RECONNECT_GATE_MAX = 3
RECONNECT_GATE_REASONS = (
    "stale_watchdog",
    "read_timeout",
    "ping_send_fail",
    "session_timeout",
)
PUBLISHER_GATE_COUNTERS = (
    "mp_try_send_full_count",
    "mp_pending_latest_replaced_count",
)


@dataclass
class PlateauRun:
    start_tick: int
    start_ts_ms: int | None
    last_tick: int
    last_ts_ms: int | None
    ticks: int


@dataclass
class PlateauMax:
    duration_ms: int | None = None
    ticks: int = 0
    start_tick: int | None = None
    end_tick: int | None = None


@dataclass
class TelemetrySummary:
    rows: int = 0
    first_tick: int | None = None
    last_tick: int | None = None
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None


@dataclass
class CapHitsSummary:
    lines: int = 0
    first_tick: int | None = None
    last_tick: int | None = None
    last_cap_hits: int | None = None
    max_cap_hits: int = 0
    total_cap_hits_est: int = 0
    max_burst: int = 0
    max_burst_from_tick: int | None = None
    max_burst_to_tick: int | None = None
    resets: int = 0


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer():
            return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if re.fullmatch(r"-?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            f = float(stripped)
        except ValueError:
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return None


def parse_venue_ids(record: dict[str, Any]) -> list[str]:
    treasury = record.get("treasury_guidance")
    if not isinstance(treasury, dict):
        return []
    venues = treasury.get("venues")
    if not isinstance(venues, list):
        return []
    mapping: dict[int, str] = {}
    for item in venues:
        if not isinstance(item, dict):
            continue
        idx = safe_int(item.get("venue_index"))
        venue_id = item.get("venue_id")
        if idx is None or not isinstance(venue_id, str):
            continue
        mapping[idx] = venue_id
    if not mapping:
        return []
    max_idx = max(mapping.keys())
    return [mapping.get(i, f"venue_{i}") for i in range(max_idx + 1)]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def fmt_ts_ms(value: int | None) -> str:
    if value is None:
        return "n/a"
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat()
    except (ValueError, OSError):
        return str(value)


def sanitize_cell(text: str) -> str:
    return text.replace("|", "\\|")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(sanitize_cell(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(sanitize_cell(c) for c in row) + " |")
    return "\n".join(out)


def parse_expected_connectors(raw: str) -> list[str]:
    connectors: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        name = token.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        connectors.append(name)
    return connectors


def update_plateau(
    active: dict[tuple[str, int], PlateauRun],
    maxima: dict[tuple[str, int], PlateauMax],
    venue: str,
    threshold_ms: int,
    tick: int,
    ts_ms: int | None,
    age_apply_ms: float | None,
) -> None:
    key = (venue, threshold_ms)
    current = active.get(key)
    above = age_apply_ms is not None and age_apply_ms > threshold_ms
    if above:
        if current is None:
            active[key] = PlateauRun(
                start_tick=tick,
                start_ts_ms=ts_ms,
                last_tick=tick,
                last_ts_ms=ts_ms,
                ticks=1,
            )
        else:
            current.last_tick = tick
            current.last_ts_ms = ts_ms
            current.ticks += 1
        return
    if current is None:
        return
    finalize_plateau_run(active, maxima, key)


def finalize_plateau_run(
    active: dict[tuple[str, int], PlateauRun],
    maxima: dict[tuple[str, int], PlateauMax],
    key: tuple[str, int],
) -> None:
    run = active.pop(key, None)
    if run is None or run.ticks < PLATEAU_MIN_TICKS:
        return

    duration_ms: int | None = None
    if (
        run.start_ts_ms is not None
        and run.last_ts_ms is not None
        and run.last_ts_ms >= run.start_ts_ms
    ):
        duration_ms = run.last_ts_ms - run.start_ts_ms

    best = maxima[key]
    should_replace = False
    if best.duration_ms is None:
        if duration_ms is not None:
            should_replace = True
        elif run.ticks > best.ticks:
            should_replace = True
    elif duration_ms is not None:
        if duration_ms > best.duration_ms:
            should_replace = True
        elif duration_ms == best.duration_ms and run.ticks > best.ticks:
            should_replace = True

    if should_replace:
        best.duration_ms = duration_ms
        best.ticks = run.ticks
        best.start_tick = run.start_tick
        best.end_tick = run.last_tick


def parse_telemetry(
    telemetry_path: Path,
) -> tuple[
    TelemetrySummary,
    dict[str, list[float]],
    dict[str, list[float]],
    dict[tuple[str, int], PlateauMax],
]:
    summary = TelemetrySummary()
    apply_values: dict[str, list[float]] = defaultdict(list)
    event_values: dict[str, list[float]] = defaultdict(list)
    known_venues: list[str] = []
    active_plateaus: dict[tuple[str, int], PlateauRun] = {}
    max_plateaus: dict[tuple[str, int], PlateauMax] = defaultdict(PlateauMax)

    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {telemetry_path}:{line_no}: {exc}") from exc
            if not isinstance(rec, dict):
                continue

            summary.rows += 1
            tick = safe_int(rec.get("t"))
            if tick is None:
                tick = summary.rows - 1

            if summary.first_tick is None:
                summary.first_tick = tick
            summary.last_tick = tick

            treasury = rec.get("treasury_guidance")
            ts_ms = safe_int(treasury.get("as_of_ms")) if isinstance(treasury, dict) else None
            if ts_ms is not None:
                if summary.first_ts_ms is None:
                    summary.first_ts_ms = ts_ms
                summary.last_ts_ms = ts_ms

            ids = parse_venue_ids(rec)
            if ids:
                if len(ids) > len(known_venues):
                    known_venues.extend(f"venue_{i}" for i in range(len(known_venues), len(ids)))
                for idx, venue in enumerate(ids):
                    known_venues[idx] = venue

            age_apply = rec.get("venue_age_ms", [])
            age_event = rec.get("venue_age_event_ms", [])
            if not isinstance(age_apply, list):
                age_apply = []
            if not isinstance(age_event, list):
                age_event = []

            venue_count = max(len(known_venues), len(age_apply), len(age_event))
            for idx in range(venue_count):
                venue = (
                    known_venues[idx]
                    if idx < len(known_venues)
                    else (ids[idx] if idx < len(ids) else f"venue_{idx}")
                )
                if idx >= len(known_venues):
                    known_venues.append(venue)

                apply_val = safe_float(age_apply[idx]) if idx < len(age_apply) else None
                if apply_val is not None and apply_val >= 0:
                    apply_values[venue].append(apply_val)
                for threshold_ms in PLATEAU_THRESHOLDS_MS:
                    update_plateau(
                        active=active_plateaus,
                        maxima=max_plateaus,
                        venue=venue,
                        threshold_ms=threshold_ms,
                        tick=tick,
                        ts_ms=ts_ms,
                        age_apply_ms=apply_val,
                    )

                event_val = safe_float(age_event[idx]) if idx < len(age_event) else None
                if event_val is not None and event_val >= 0:
                    event_values[venue].append(event_val)

    for key in list(active_plateaus.keys()):
        finalize_plateau_run(active_plateaus, max_plateaus, key)

    return summary, apply_values, event_values, max_plateaus


WS_AUDIT_RECONNECT_RE = re.compile(
    r"WS_AUDIT\s+venue=(?P<venue>[a-zA-Z0-9_]+)\s+reconnect_reason=(?P<reason>[a-zA-Z0-9_]+)\s+count=(?P<count>\d+)"
)
MP_COUNTER_RE = re.compile(r"\b(mp_[a-z0-9_]*_count)=(\d+)\b")


def infer_venue_from_line(lower_line: str) -> str | None:
    for venue in VENUE_HINTS:
        if venue in lower_line:
            return venue
    return None


def infer_reason_from_line(lower_line: str) -> str | None:
    if "watchdog" in lower_line and "reconnect" in lower_line:
        return "stale_watchdog"
    if "ping send failed" in lower_line and "reconnect" in lower_line:
        return "ping_send_fail"
    if "read timeout" in lower_line and "reconnect" in lower_line:
        return "read_timeout"
    if "session timeout" in lower_line and "reconnect" in lower_line:
        return "session_timeout"
    if "subscribe error" in lower_line:
        return "subscribe_error"
    if "too many parse errors" in lower_line and "reconnect" in lower_line:
        return "parse_error"
    if "forcing reconnect for fresh snapshot" in lower_line:
        return "decode_fail_loop"
    if "seq_gap" in lower_line or "seq gap" in lower_line:
        return "seq_gap"
    if "seq_mismatch" in lower_line or "seq mismatch" in lower_line:
        return "seq_mismatch"
    if "ws closed; reconnecting" in lower_line:
        return "ws_closed"
    if "stream ended" in lower_line and "ws" in lower_line:
        return "stream_ended"
    if "connect timeout" in lower_line:
        return "connect_timeout"
    if "connect error" in lower_line:
        return "connect_error"
    return None


def parse_run_log(
    run_log_path: Path,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int], dict[str, int]]:
    audit_reconnect_counts: dict[tuple[str, str], int] = defaultdict(int)
    signature_reconnect_counts: dict[tuple[str, str], int] = defaultdict(int)
    market_publisher_counters: dict[str, int] = defaultdict(int)

    with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "WS_AUDIT" in line:
                reconnect_match = WS_AUDIT_RECONNECT_RE.search(line)
                if reconnect_match:
                    venue = reconnect_match.group("venue").lower()
                    reason = reconnect_match.group("reason")
                    count = int(reconnect_match.group("count"))
                    key = (venue, reason)
                    if count > audit_reconnect_counts[key]:
                        audit_reconnect_counts[key] = count

                if "component=market_publisher" in line:
                    for name, value in MP_COUNTER_RE.findall(line):
                        count = safe_int(value)
                        if count is None:
                            continue
                        if count > market_publisher_counters[name]:
                            market_publisher_counters[name] = count

            lower = line.lower()
            reason = infer_reason_from_line(lower)
            if reason is None:
                continue
            venue = infer_venue_from_line(lower) or "unknown"
            signature_reconnect_counts[(venue, reason)] += 1

    return audit_reconnect_counts, signature_reconnect_counts, market_publisher_counters


KV_TOKEN_RE = re.compile(r"\b([a-zA-Z0-9_]+)=([^\s]+)\b")


def parse_market_rx_stats(path: Path) -> CapHitsSummary:
    summary = CapHitsSummary()
    prev_cap_hits: int | None = None
    prev_tick: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "market_rx_stats" not in line:
                continue
            summary.lines += 1
            pairs = {k: v for (k, v) in KV_TOKEN_RE.findall(line)}
            tick = safe_int(pairs.get("tick"))
            cap_hits = safe_int(pairs.get("cap_hits"))
            if cap_hits is None:
                continue

            if tick is not None:
                if summary.first_tick is None:
                    summary.first_tick = tick
                summary.last_tick = tick

            summary.max_cap_hits = max(summary.max_cap_hits, cap_hits)

            if prev_cap_hits is None:
                summary.total_cap_hits_est += max(cap_hits, 0)
            else:
                if cap_hits >= prev_cap_hits:
                    delta = cap_hits - prev_cap_hits
                else:
                    summary.resets += 1
                    delta = cap_hits
                summary.total_cap_hits_est += max(delta, 0)
                if delta > summary.max_burst:
                    summary.max_burst = delta
                    summary.max_burst_from_tick = prev_tick
                    summary.max_burst_to_tick = tick

            prev_cap_hits = cap_hits
            prev_tick = tick
            summary.last_cap_hits = cap_hits

    return summary


def combined_reconnect_count(
    audit_reconnect: dict[tuple[str, str], int],
    signature_reconnect: dict[tuple[str, str], int],
    venue: str,
    reason: str,
) -> int:
    audit_count = audit_reconnect.get((venue, reason), 0)
    signature_count = signature_reconnect.get((venue, reason), 0)
    combined = audit_count if audit_count > 0 else signature_count
    if audit_count > 0 and signature_count > audit_count:
        combined = signature_count
    return combined


def evaluate_frontier_gate(
    apply_values: dict[str, list[float]],
    event_values: dict[str, list[float]],
    max_plateaus: dict[tuple[str, int], PlateauMax],
    audit_reconnect: dict[tuple[str, str], int],
    signature_reconnect: dict[tuple[str, str], int],
    market_publisher_counters: dict[str, int],
    cap_hits_summary: CapHitsSummary | None,
    expected_connectors: list[str],
    require_event_age: bool,
) -> list[str]:
    failures: list[str] = []

    for venue in expected_connectors:
        apply = apply_values.get(venue, [])
        if not apply:
            failures.append(f"missing apply-age coverage for venue '{venue}' (apply_n=0)")
            continue
        apply_p95 = percentile(apply, 95.0)
        apply_p99 = percentile(apply, 99.0)
        if apply_p95 is not None and apply_p95 > APPLY_P95_MAX_MS:
            failures.append(
                f"apply-age p95 above threshold for venue '{venue}' ({apply_p95:.1f}ms > {APPLY_P95_MAX_MS:.0f}ms)"
            )
        if apply_p99 is not None and apply_p99 > APPLY_P99_MAX_MS:
            failures.append(
                f"apply-age p99 above threshold for venue '{venue}' ({apply_p99:.1f}ms > {APPLY_P99_MAX_MS:.0f}ms)"
            )

        plateau = max_plateaus.get((venue, PLATEAU_GATE_THRESHOLD_MS), PlateauMax())
        if plateau.duration_ms is not None and plateau.duration_ms > 0:
            failures.append(
                f"stale plateau at {PLATEAU_GATE_THRESHOLD_MS}ms for venue '{venue}' (max_duration_s={plateau.duration_ms / 1000.0:.2f})"
            )
        elif plateau.duration_ms is None and plateau.ticks > 0:
            failures.append(
                f"stale plateau at {PLATEAU_GATE_THRESHOLD_MS}ms for venue '{venue}' has non-zero ticks without duration"
            )

        for reason in RECONNECT_GATE_REASONS:
            combined = combined_reconnect_count(
                audit_reconnect=audit_reconnect,
                signature_reconnect=signature_reconnect,
                venue=venue,
                reason=reason,
            )
            if combined > RECONNECT_GATE_MAX:
                failures.append(
                    f"reconnect threshold exceeded for venue '{venue}', reason '{reason}' (combined={combined} > {RECONNECT_GATE_MAX})"
                )

    if require_event_age:
        for venue in expected_connectors:
            event = event_values.get(venue, [])
            if not event:
                failures.append(f"missing event-age coverage for venue '{venue}' (event_n=0)")
                continue
            event_p95 = percentile(event, 95.0)
            event_p99 = percentile(event, 99.0)
            if event_p95 is not None and event_p95 > EVENT_P95_MAX_MS:
                failures.append(
                    f"event-age p95 above threshold for venue '{venue}' ({event_p95:.1f}ms > {EVENT_P95_MAX_MS:.0f}ms)"
                )
            if event_p99 is not None and event_p99 > EVENT_P99_MAX_MS:
                failures.append(
                    f"event-age p99 above threshold for venue '{venue}' ({event_p99:.1f}ms > {EVENT_P99_MAX_MS:.0f}ms)"
                )

    for name in PUBLISHER_GATE_COUNTERS:
        value = market_publisher_counters.get(name)
        if value is None:
            failures.append(f"missing market_publisher counter '{name}'")
        elif value != 0:
            failures.append(f"market_publisher counter '{name}' is non-zero ({value})")

    if cap_hits_summary is None:
        failures.append("missing market_rx_stats.log (cap_hits evidence unavailable)")
    else:
        if cap_hits_summary.total_cap_hits_est != 0:
            failures.append(
                f"runner cap_hits total is non-zero ({cap_hits_summary.total_cap_hits_est})"
            )
        if cap_hits_summary.max_burst != 0:
            failures.append(f"runner cap_hits max burst is non-zero (+{cap_hits_summary.max_burst})")

    return failures


def build_report(
    out_dir: Path,
    telemetry_summary: TelemetrySummary,
    apply_values: dict[str, list[float]],
    event_values: dict[str, list[float]],
    max_plateaus: dict[tuple[str, int], PlateauMax],
    audit_reconnect: dict[tuple[str, str], int],
    signature_reconnect: dict[tuple[str, str], int],
    market_publisher_counters: dict[str, int],
    cap_hits_summary: CapHitsSummary | None,
) -> str:
    lines: list[str] = []
    lines.append("# WS Shadow Soak Report")
    lines.append("")
    lines.append("## Run Inputs")
    lines.append(f"- out_dir: `{out_dir}`")
    lines.append(f"- telemetry rows: `{telemetry_summary.rows}`")
    lines.append(
        "- tick range: "
        f"`{fmt_int(telemetry_summary.first_tick)} -> {fmt_int(telemetry_summary.last_tick)}`"
    )
    lines.append(
        "- treasury as_of range (UTC): "
        f"`{fmt_ts_ms(telemetry_summary.first_ts_ms)} -> {fmt_ts_ms(telemetry_summary.last_ts_ms)}`"
    )
    lines.append("")

    lines.append("## Venue Age Percentiles (ms)")
    age_rows: list[list[str]] = []
    venues = sorted(set(apply_values.keys()) | set(event_values.keys()))
    for venue in venues:
        apply = apply_values.get(venue, [])
        event = event_values.get(venue, [])
        age_rows.append(
            [
                venue,
                str(len(apply)),
                fmt_ms(percentile(apply, 50.0)),
                fmt_ms(percentile(apply, 95.0)),
                fmt_ms(percentile(apply, 99.0)),
                str(len(event)),
                fmt_ms(percentile(event, 50.0)),
                fmt_ms(percentile(event, 95.0)),
                fmt_ms(percentile(event, 99.0)),
            ]
        )
    if age_rows:
        lines.append(
            md_table(
                [
                    "venue",
                    "apply_n",
                    "apply_p50",
                    "apply_p95",
                    "apply_p99",
                    "event_n",
                    "event_p50",
                    "event_p95",
                    "event_p99",
                ],
                age_rows,
            )
        )
    else:
        lines.append("_No venue age samples found._")
    lines.append("")

    lines.append("## Reconnect Reason Counts")
    reconnect_rows: list[list[str]] = []
    reconnect_keys = sorted(set(audit_reconnect.keys()) | set(signature_reconnect.keys()))
    for venue, reason in reconnect_keys:
        audit_count = audit_reconnect.get((venue, reason), 0)
        signature_count = signature_reconnect.get((venue, reason), 0)
        combined = audit_count if audit_count > 0 else signature_count
        if audit_count > 0 and signature_count > audit_count:
            combined = signature_count
        reconnect_rows.append(
            [
                venue,
                reason,
                str(audit_count),
                str(signature_count),
                str(combined),
            ]
        )
    if reconnect_rows:
        lines.append(
            md_table(
                ["venue", "reason", "ws_audit_count", "signature_count", "combined_count"],
                reconnect_rows,
            )
        )
    else:
        lines.append("_No reconnect evidence found in run.log._")
    lines.append("")

    lines.append("## MarketPublisher Pressure Counters (WS_AUDIT)")
    if market_publisher_counters:
        rows = [[name, str(value)] for name, value in sorted(market_publisher_counters.items())]
        lines.append(md_table(["counter", "max_count"], rows))
    else:
        lines.append("_No `component=market_publisher` counters found in run.log._")
    lines.append("")

    lines.append("## Runner cap_hits Summary")
    if cap_hits_summary is None:
        lines.append("_market_rx_stats.log not present in out-dir._")
    else:
        lines.append(f"- parsed lines: `{cap_hits_summary.lines}`")
        lines.append(
            f"- tick range: `{fmt_int(cap_hits_summary.first_tick)} -> {fmt_int(cap_hits_summary.last_tick)}`"
        )
        lines.append(f"- last cap_hits: `{fmt_int(cap_hits_summary.last_cap_hits)}`")
        lines.append(f"- max observed cap_hits: `{cap_hits_summary.max_cap_hits}`")
        lines.append(f"- estimated total cap_hits: `{cap_hits_summary.total_cap_hits_est}`")
        lines.append(
            "- worst interval burst: "
            f"`+{cap_hits_summary.max_burst}` "
            f"(ticks `{fmt_int(cap_hits_summary.max_burst_from_tick)} -> {fmt_int(cap_hits_summary.max_burst_to_tick)}`)"
        )
        lines.append(f"- counter resets detected: `{cap_hits_summary.resets}`")
    lines.append("")

    lines.append(
        "## Max Stale Plateau Durations (apply-age `venue_age_ms`)"
    )
    lines.append(
        f"_Plateau definition: `age_apply_ms > threshold` for at least `{PLATEAU_MIN_TICKS}` consecutive ticks._"
    )
    plateau_rows: list[list[str]] = []
    for venue in venues:
        for threshold_ms in PLATEAU_THRESHOLDS_MS:
            plateau = max_plateaus.get((venue, threshold_ms), PlateauMax())
            duration_s = (
                f"{plateau.duration_ms / 1000.0:.2f}" if plateau.duration_ms is not None else "n/a"
            )
            plateau_rows.append(
                [
                    venue,
                    str(threshold_ms),
                    duration_s,
                    str(plateau.ticks),
                    fmt_int(plateau.start_tick),
                    fmt_int(plateau.end_tick),
                ]
            )
    if plateau_rows:
        lines.append(
            md_table(
                [
                    "venue",
                    "threshold_ms",
                    "max_duration_s",
                    "ticks",
                    "start_tick",
                    "end_tick",
                ],
                plateau_rows,
            )
        )
    else:
        lines.append("_No plateau data found._")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a markdown WS soak report from run artifacts.")
    parser.add_argument("--out-dir", required=True, help="Run output directory containing telemetry.jsonl and run.log")
    parser.add_argument("--gate", action="store_true", help="Evaluate frontier readiness gate and return non-zero on failure.")
    parser.add_argument(
        "--expected-connectors",
        default="",
        help="Comma-separated connector list used for venue coverage/tail checks.",
    )
    parser.add_argument(
        "--require-event-age",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require event-age coverage and thresholds when gate is enabled (default: true).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    telemetry_path = out_dir / "telemetry.jsonl"
    run_log_path = out_dir / "run.log"
    market_rx_path = out_dir / "market_rx_stats.log"

    if not telemetry_path.exists():
        print(f"error: missing required file: {telemetry_path}", file=sys.stderr)
        return 2
    if not run_log_path.exists():
        print(f"error: missing required file: {run_log_path}", file=sys.stderr)
        return 2

    try:
        telemetry_summary, apply_values, event_values, max_plateaus = parse_telemetry(telemetry_path)
        audit_reconnect, signature_reconnect, market_publisher = parse_run_log(run_log_path)
        cap_hits = parse_market_rx_stats(market_rx_path) if market_rx_path.exists() else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = build_report(
        out_dir=out_dir,
        telemetry_summary=telemetry_summary,
        apply_values=apply_values,
        event_values=event_values,
        max_plateaus=max_plateaus,
        audit_reconnect=audit_reconnect,
        signature_reconnect=signature_reconnect,
        market_publisher_counters=market_publisher,
        cap_hits_summary=cap_hits,
    )

    report_path = out_dir / "ws_soak_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\n_report saved to `{report_path}`_")

    if args.gate:
        expected_connectors = parse_expected_connectors(args.expected_connectors)
        gate_failures = evaluate_frontier_gate(
            apply_values={k.lower(): v for k, v in apply_values.items()},
            event_values={k.lower(): v for k, v in event_values.items()},
            max_plateaus={(venue.lower(), threshold): plateau for (venue, threshold), plateau in max_plateaus.items()},
            audit_reconnect={(venue.lower(), reason.lower()): count for (venue, reason), count in audit_reconnect.items()},
            signature_reconnect={
                (venue.lower(), reason.lower()): count for (venue, reason), count in signature_reconnect.items()
            },
            market_publisher_counters=market_publisher,
            cap_hits_summary=cap_hits,
            expected_connectors=expected_connectors,
            require_event_age=args.require_event_age,
        )
        if gate_failures:
            print("GATE: FAIL")
            for reason in gate_failures:
                print(f"  - {reason}")
            return 2
        print("GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
