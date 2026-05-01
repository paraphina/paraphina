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
EXTENDED_DEFECT_CLASSES = (
    "no_data_transport_gap",
    "data_seen_no_publish",
    "runner_freeze_apply_gap",
    "future_timestamp_deferral",
    "unclassified",
)
EXTENDED_BOOTSTRAP_CLASSES = (
    "bootstrap_no_first_frame",
    "bootstrap_frame_no_book",
    "bootstrap_book_no_publish",
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


def classify_extended_stale_episode(
    ws_msg_pairs: dict[str, Any], runner_truth_pairs: dict[str, Any] | None
) -> str:
    stale_ms = safe_float(ws_msg_pairs.get("stale_ms")) or 0.0
    half_stale_ms = stale_ms / 2.0 if stale_ms > 0 else 0.0
    age_ws_rx_ms = safe_float(ws_msg_pairs.get("age_ws_rx_ms"))
    age_data_rx_ms = safe_float(ws_msg_pairs.get("age_data_rx_ms"))
    age_book_event_ms = safe_float(ws_msg_pairs.get("age_book_event_ms"))
    age_published_ms = safe_float(ws_msg_pairs.get("age_published_ms"))

    if (
        age_ws_rx_ms is not None
        and age_data_rx_ms is not None
        and age_book_event_ms is not None
        and age_published_ms is not None
        and age_ws_rx_ms < half_stale_ms
        and age_data_rx_ms >= stale_ms
        and age_book_event_ms >= stale_ms
        and age_published_ms >= stale_ms
    ):
        return "no_data_transport_gap"

    if (
        age_data_rx_ms is not None
        and age_book_event_ms is not None
        and age_published_ms is not None
        and age_data_rx_ms < half_stale_ms
        and (age_book_event_ms >= stale_ms or age_published_ms >= stale_ms)
    ):
        return "data_seen_no_publish"

    if runner_truth_pairs:
        age_apply_ms = safe_float(runner_truth_pairs.get("age_apply_ms"))
        age_event_ms = safe_float(runner_truth_pairs.get("age_event_ms"))
        venue_state_stale_ms = safe_float(runner_truth_pairs.get("venue_state_stale_ms"))
        ext_apply_frozen_total = safe_int(runner_truth_pairs.get("ext_apply_frozen_total")) or 0
        ext_future_total = safe_int(runner_truth_pairs.get("ext_future_total")) or 0
        freeze_warning_count = safe_int(runner_truth_pairs.get("_freeze_warning_count")) or 0

        if (
            age_published_ms is not None
            and age_apply_ms is not None
            and venue_state_stale_ms is not None
            and age_published_ms < half_stale_ms
            and age_apply_ms >= venue_state_stale_ms
            and (ext_apply_frozen_total > 0 or freeze_warning_count > 0)
        ):
            return "runner_freeze_apply_gap"

        if (
            age_event_ms is not None
            and age_apply_ms is not None
            and venue_state_stale_ms is not None
            and ext_future_total > 0
            and age_event_ms < half_stale_ms
            and age_apply_ms >= venue_state_stale_ms
        ):
            return "future_timestamp_deferral"

    return "unclassified"


def summarize_extended_defects(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {name: 0 for name in EXTENDED_DEFECT_CLASSES}
    for episode in episodes:
        defect_class = str(episode.get("defect_class") or "unclassified")
        counts.setdefault(defect_class, 0)
        counts[defect_class] += 1

    total = sum(counts.values())
    dominant_class = "unclassified"
    dominant_count = 0
    if total > 0:
        dominant_class, dominant_count = max(counts.items(), key=lambda item: item[1])
    confidence_pct = (dominant_count / total * 100.0) if total > 0 else 0.0
    return {
        "counts": counts,
        "total": total,
        "dominant_class": dominant_class,
        "dominant_count": dominant_count,
        "confidence_pct": confidence_pct,
    }


def summarize_extended_bootstrap(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {name: 0 for name in EXTENDED_BOOTSTRAP_CLASSES}
    for episode in episodes:
        reason = str(episode.get("reason") or "bootstrap_no_first_frame")
        counts.setdefault(reason, 0)
        counts[reason] += 1

    total = sum(counts.values())
    dominant_reason = "bootstrap_no_first_frame"
    dominant_count = 0
    if total > 0:
        dominant_reason, dominant_count = max(counts.items(), key=lambda item: item[1])
    confidence_pct = (dominant_count / total * 100.0) if total > 0 else 0.0
    return {
        "counts": counts,
        "total": total,
        "dominant_reason": dominant_reason,
        "dominant_count": dominant_count,
        "confidence_pct": confidence_pct,
    }


def summarize_extended_bootstrap_stages(
    timeout_stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    counts = {"first_frame": 0, "post_first_frame": 0}
    for reason, stats in timeout_stats.items():
        samples = safe_int(stats.get("samples")) or 0
        stage = str(
            stats.get("last_bootstrap_timeout_stage")
            or ("first_frame" if reason == "bootstrap_no_first_frame" else "post_first_frame")
        )
        counts.setdefault(stage, 0)
        counts[stage] += samples

    total = sum(counts.values())
    dominant_stage = "first_frame"
    dominant_count = 0
    if total > 0:
        dominant_stage, dominant_count = max(counts.items(), key=lambda item: item[1])
    confidence_pct = (dominant_count / total * 100.0) if total > 0 else 0.0
    return {
        "counts": counts,
        "total": total,
        "dominant_stage": dominant_stage,
        "dominant_count": dominant_count,
        "confidence_pct": confidence_pct,
    }


def summarize_extended_first_frame_timeout(
    timeout_stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    stats = dict(timeout_stats.get("bootstrap_no_first_frame", {}))
    return {
        "samples": safe_int(stats.get("samples")) or 0,
        "last_connect_first_frame_timeout_ms": safe_int(
            stats.get("last_connect_first_frame_timeout_ms")
        ),
        "max_connect_first_frame_timeout_ms": safe_int(
            stats.get("max_connect_first_frame_timeout_ms")
        ),
        "last_stale_watchdog_deferred_until_first_publish": safe_int(
            stats.get("last_stale_watchdog_deferred_until_first_publish")
        ),
        "last_stale_watchdog_armed": safe_int(stats.get("last_stale_watchdog_armed")),
        "max_time_to_first_message_ms": safe_int(
            stats.get("max_time_to_first_message_ms")
        ),
    }


def summarize_extended_first_data_timeout(
    timeout_stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    stats = dict(timeout_stats.get("bootstrap_no_first_frame", {}))
    return {
        "samples": safe_int(stats.get("samples")) or 0,
        "last_connect_first_frame_timeout_ms": safe_int(
            stats.get("last_connect_first_frame_timeout_ms")
        ),
        "max_connect_first_frame_timeout_ms": safe_int(
            stats.get("max_connect_first_frame_timeout_ms")
        ),
        "last_first_control_frame_seen": safe_int(
            stats.get("last_first_control_frame_seen")
        ),
        "last_first_control_frame_kind": stats.get("last_first_control_frame_kind"),
        "last_first_data_frame_seen": safe_int(stats.get("last_first_data_frame_seen")),
        "last_rest_seed_bridge_active": safe_int(
            stats.get("last_rest_seed_bridge_active")
        ),
        "last_stale_watchdog_deferred_until_first_publish": safe_int(
            stats.get("last_stale_watchdog_deferred_until_first_publish")
        ),
        "max_time_to_first_control_frame_ms": safe_int(
            stats.get("max_time_to_first_control_frame_ms")
        ),
        "max_time_to_first_message_ms": safe_int(
            stats.get("max_time_to_first_message_ms")
        ),
    }


def summarize_extended_watchdog_bootstrap_transition(
    transport_gap_stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    stats = dict(transport_gap_stats.get("extended", {}))
    return {
        "samples": safe_int(stats.get("watchdog_bootstrap_transition_samples")) or 0,
        "last_first_publish_observed": safe_int(stats.get("last_first_publish_observed")),
        "last_watchdog_armed_now": safe_int(stats.get("last_watchdog_armed_now")),
        "last_stale_watchdog_deferred_until_first_publish": safe_int(
            stats.get("last_stale_watchdog_deferred_until_first_publish")
        ),
        "last_time_to_first_publish_ms": safe_int(
            stats.get("last_time_to_first_publish_ms")
        ),
        "max_time_to_first_publish_ms": safe_int(
            stats.get("max_time_to_first_publish_ms")
        ),
    }


def summarize_extended_rest_seed(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    failures_by_status: dict[str, int] = {}
    failures_by_http_status: dict[str, int] = {}
    for key, value in list(summary.items()):
        if key.startswith("status_") and key.endswith("_count"):
            status = key[len("status_") : -len("_count")]
            if status != "ok":
                failures_by_status[status] = safe_int(value) or 0
        if key.startswith("http_status_") and key.endswith("_count"):
            status = key[len("http_status_") : -len("_count")]
            failures_by_http_status[status] = safe_int(value) or 0
    return {
        "extended": summary,
        "failures_by_status": failures_by_status,
        "failures_by_http_status": failures_by_http_status,
    }


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


def parse_kv_tokens(line: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key:
            continue
        pairs[key] = value.rstrip(",")
    return pairs


def update_max_stat(stats: dict[str, Any], key: str, value: int | None) -> None:
    if value is None:
        return
    prev = safe_int(stats.get(key))
    if prev is None or value > prev:
        stats[key] = value


def update_max_numeric_token(stats: dict[str, Any], key: str, raw: str | None) -> None:
    if raw is None:
        return
    int_value = safe_int(raw)
    field = f"max_{key}"
    if int_value is not None:
        prev = safe_int(stats.get(field))
        if prev is None or int_value > prev:
            stats[field] = int_value
        return
    float_value = safe_float(raw)
    if float_value is None:
        return
    prev = safe_float(stats.get(field))
    if prev is None or float_value > prev:
        stats[field] = float_value


def infer_venue_from_line(lower_line: str) -> str | None:
    for venue in VENUE_HINTS:
        if venue in lower_line:
            return venue
    return None


def infer_reason_from_line(lower_line: str) -> str | None:
    if "bootstrap no first frame" in lower_line:
        return "bootstrap_no_first_frame"
    if "bootstrap frame/no-book" in lower_line:
        return "bootstrap_frame_no_book"
    if "bootstrap book/no-publish" in lower_line:
        return "bootstrap_book_no_publish"
    if "post publish transport gap" in lower_line:
        return "post_publish_transport_gap"
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
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    audit_reconnect_counts: dict[tuple[str, str], int] = defaultdict(int)
    signature_reconnect_counts: dict[tuple[str, str], int] = defaultdict(int)
    market_publisher_counters: dict[str, int] = defaultdict(int)
    runner_apply_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    runner_apply_truth_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    rest_monitor_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    arb_gate_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    hl_pubq_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_ws_msg_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_cfg_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_reconnect_policy_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_transport_gap_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_rest_seed_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_seed_bridge_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_control_frame_grace_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_session_hedge_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_backend_attach_fallback_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_post_publish_fallback_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_bootstrap_timeout_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_bootstrap_churn_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_socket_establishment_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_socket_role_progress_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    extended_stream_kind_progress_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    ping_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    lighter_ts_fallback_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    aster_book_recovery_stats: dict[str, dict[str, Any]] = defaultdict(dict)
    latest_extended_runner_truth: dict[str, Any] | None = None
    extended_defect_episodes: list[dict[str, Any]] = []
    extended_bootstrap_episodes: list[dict[str, Any]] = []

    with run_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "WS_AUDIT" in line:
                pairs = parse_kv_tokens(line)
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

                if "component=runner_apply_truth" in line:
                    venue = str(pairs.get("venue", "unknown")).lower()
                    stats = runner_apply_truth_stats[venue]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)
                    for field in (
                        "age_apply_ms",
                        "age_event_ms",
                        "last_candidate_mid",
                        "last_candidate_spread",
                        "last_prev_mid",
                        "last_prev_spread",
                        "venue_state_stale_ms",
                    ):
                        value = safe_float(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                    if venue == "extended":
                        latest_extended_runner_truth = dict(pairs)
                        freeze_warning_count = (
                            safe_int(stats.get("freeze_warning_count")) or 0
                        )
                        if freeze_warning_count > 0:
                            latest_extended_runner_truth["_freeze_warning_count"] = str(
                                freeze_warning_count
                            )

                if "WARN: Extended core book update frozen" in line:
                    stats = runner_apply_truth_stats["extended"]
                    freeze_warning_count = (
                        safe_int(stats.get("freeze_warning_count")) or 0
                    ) + 1
                    stats["freeze_warning_count"] = freeze_warning_count
                    update_max_stat(
                        stats,
                        "max_freeze_warning_count",
                        freeze_warning_count,
                    )
                    if latest_extended_runner_truth is None:
                        latest_extended_runner_truth = {}
                    latest_extended_runner_truth["_freeze_warning_count"] = str(
                        freeze_warning_count
                    )

                if pairs.get("component") == "runner_apply":
                    venue = str(pairs.get("venue", "unknown")).lower()
                    stats = runner_apply_stats[venue]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    update_max_stat(stats, "max_cache_err", safe_int(pairs.get("cache_err")))
                    update_max_stat(stats, "max_ext_future", safe_int(pairs.get("ext_future")))
                    age_apply = safe_float(pairs.get("age_apply_ms"))
                    if age_apply is not None:
                        stats["last_age_apply_ms"] = age_apply
                    age_event = safe_float(pairs.get("age_event_ms"))
                    if age_event is not None:
                        stats["last_age_event_ms"] = age_event

                if "subsystem=rest_monitor" in line:
                    venue = str(pairs.get("venue", "unknown")).lower()
                    stats = rest_monitor_stats[venue]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    for field in (
                        "rest_check_count",
                        "rest_attempt_count",
                        "rest_success_count",
                        "rest_fail_count",
                        "rest_inject_count",
                        "rest_suppressed_count",
                    ):
                        update_max_stat(stats, f"max_{field}", safe_int(pairs.get(field)))
                    age_ms = safe_float(pairs.get("age_ms"))
                    if age_ms is not None:
                        stats["last_age_ms"] = age_ms
                    threshold_ms = safe_int(pairs.get("threshold_ms"))
                    if threshold_ms is not None:
                        stats["last_threshold_ms"] = threshold_ms

                if "subsystem=arb_gate" in line:
                    venue = str(pairs.get("venue", "unknown")).lower()
                    stats = arb_gate_stats[venue]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    update_max_stat(stats, "max_gated_ticks", safe_int(pairs.get("gated_ticks")))
                    last_apply_age_ms = safe_int(pairs.get("last_apply_age_ms"))
                    if last_apply_age_ms is not None:
                        stats["last_apply_age_ms"] = last_apply_age_ms
                    threshold_ms = safe_int(pairs.get("threshold_ms"))
                    if threshold_ms is not None:
                        stats["threshold_ms"] = threshold_ms

                if "component=hl_pubq" in line:
                    venue = str(pairs.get("venue", "unknown")).lower()
                    stats = hl_pubq_stats[venue]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    for field in (
                        "try_send_full",
                        "pending_overwrite",
                        "pending_lock_fail",
                        "ts_zero_count",
                        "ws_rx_age_ms",
                        "data_rx_age_ms",
                        "pub_age_ms",
                        "book_age_ms",
                        "pub_minus_book_age_ms",
                        "send_block_max_ms",
                        "send_block_gt_5ms",
                        "send_block_gt_50ms",
                        "send_block_gt_250ms",
                        "forward_send_count",
                        "forward_send_err_count",
                        "coalesced_drop_count",
                        "pending_take_count",
                        "ts_missing_or_zero_count",
                        "ts_clamped_past_skew_count",
                        "ts_clamped_future_skew_count",
                        "ts_policy_enabled",
                        "ts_policy_applied_count",
                        "ts_kept_exchange_count",
                        "ts_past_skew_max_ms",
                        "ts_future_skew_max_ms",
                        "queued_hiwater",
                        "queued_len",
                    ):
                        update_max_stat(stats, f"max_{field}", safe_int(pairs.get(field)))

                if "component=ws_msg" in line and "venue=extended" in line:
                    stats = extended_ws_msg_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    reason = pairs.get("reason")
                    if reason:
                        stats["last_reason"] = reason
                    for field in ("last_frame_kind", "last_data_kind"):
                        value = pairs.get(field)
                        if value:
                            stats[f"last_{field}"] = value
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)
                    if reason in {
                        "stale_watchdog",
                        "read_timeout",
                        "post_publish_transport_gap",
                    }:
                        extended_defect_episodes.append(
                            {
                                "reason": reason,
                                "defect_class": classify_extended_stale_episode(
                                    pairs, latest_extended_runner_truth
                                ),
                                "age_ws_rx_ms": safe_float(pairs.get("age_ws_rx_ms")),
                                "age_data_rx_ms": safe_float(pairs.get("age_data_rx_ms")),
                                "age_book_event_ms": safe_float(
                                    pairs.get("age_book_event_ms")
                                ),
                                "age_published_ms": safe_float(
                                    pairs.get("age_published_ms")
                                ),
                            }
                        )

                if "extended_read_timeout_ms=" in line and "venue=extended" in line:
                    stats = extended_cfg_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    timeout_ms = safe_int(pairs.get("extended_read_timeout_ms"))
                    if timeout_ms is not None:
                        stats["last_extended_read_timeout_ms"] = timeout_ms
                    connect_first_frame_timeout_ms = safe_int(
                        pairs.get("extended_connect_first_frame_timeout_ms")
                    )
                    if connect_first_frame_timeout_ms is not None:
                        stats[
                            "last_extended_connect_first_frame_timeout_ms"
                        ] = connect_first_frame_timeout_ms
                    connect_book_timeout_ms = safe_int(
                        pairs.get("extended_connect_book_timeout_ms")
                    )
                    if connect_book_timeout_ms is not None:
                        stats[
                            "last_extended_connect_book_timeout_ms"
                        ] = connect_book_timeout_ms
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=reconnect_policy" in line and "venue=extended" in line:
                    reason = str(pairs.get("reason", "unknown")).lower()
                    stats = extended_reconnect_policy_stats[reason]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    stats["reason"] = reason
                    sleep_ms = safe_int(pairs.get("sleep_ms"))
                    if sleep_ms is not None:
                        stats["last_sleep_ms"] = sleep_ms
                        update_max_stat(stats, "max_sleep_ms", sleep_ms)
                    suppressed = safe_int(pairs.get("failure_escalation_suppressed"))
                    if suppressed is not None:
                        stats["last_failure_escalation_suppressed"] = suppressed
                        update_max_stat(
                            stats,
                            "max_failure_escalation_suppressed",
                            suppressed,
                        )
                    consecutive_failures = safe_int(pairs.get("consecutive_failures"))
                    if consecutive_failures is not None:
                        stats["last_consecutive_failures"] = consecutive_failures
                        update_max_stat(
                            stats, "max_consecutive_failures", consecutive_failures
                        )
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=rest_snapshot_seed" in line and "venue=extended" in line:
                    stats = extended_rest_seed_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    status = str(pairs.get("status", "unknown")).lower()
                    stats["last_status"] = status
                    stats[f"status_{status}_count"] = (
                        safe_int(stats.get(f"status_{status}_count")) or 0
                    ) + 1
                    http_status = safe_int(pairs.get("http_status"))
                    if http_status is not None:
                        stats["last_http_status"] = http_status
                        stats[f"http_status_{http_status}_count"] = (
                            safe_int(stats.get(f"http_status_{http_status}_count")) or 0
                        ) + 1
                    endpoint_kind = pairs.get("endpoint_kind")
                    if endpoint_kind:
                        stats["last_endpoint_kind"] = endpoint_kind
                    market = pairs.get("market")
                    if market:
                        stats["last_market"] = market
                    for field in ("latency_ms", "seeded", "bid_levels", "ask_levels"):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=socket_establishment" in line and "venue=extended" in line:
                    stats = extended_socket_establishment_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    socket_role = str(pairs.get("socket_role", "unknown")).lower()
                    stream_kind = str(pairs.get("stream_kind", "unknown")).lower()
                    stats["last_action"] = action
                    stats["last_socket_role"] = socket_role
                    stats["last_stream_kind"] = stream_kind
                    stats[f"action_{action}_count"] = (
                        safe_int(stats.get(f"action_{action}_count")) or 0
                    ) + 1
                    stats[f"{socket_role}_{action}_count"] = (
                        safe_int(stats.get(f"{socket_role}_{action}_count")) or 0
                    ) + 1
                    stats[f"{stream_kind}_{action}_count"] = (
                        safe_int(stats.get(f"{stream_kind}_{action}_count")) or 0
                    ) + 1
                    for field in ("host", "path", "failure_stage", "failure_class"):
                        value = pairs.get(field)
                        if value:
                            stats[f"last_{field}"] = value
                    for field in (
                        "disable_nagle",
                        "elapsed_ms",
                        "tcp_connect_ms",
                        "ws_upgrade_ms",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=bootstrap_seed_bridge" in line and "venue=extended" in line:
                    stats = extended_seed_bridge_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    stats[f"action_{action}_count"] = (
                        safe_int(stats.get(f"action_{action}_count")) or 0
                    ) + 1
                    clear_reason = pairs.get("clear_reason")
                    if clear_reason:
                        stats["last_clear_reason"] = clear_reason
                    for field in (
                        "rest_snapshot_seeded",
                        "rest_seed_bridge_active",
                        "seed_age_ms",
                        "venue_state_stale_ms",
                        "connect_first_frame_timeout_ms",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if (
                    "component=bootstrap_control_frame_grace" in line
                    and "venue=extended" in line
                ):
                    stats = extended_control_frame_grace_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    stats[f"action_{action}_count"] = (
                        safe_int(stats.get(f"action_{action}_count")) or 0
                    ) + 1
                    reason_family = pairs.get("reason_family")
                    if reason_family:
                        stats["last_reason_family"] = reason_family
                    first_control_frame_kind = pairs.get("first_control_frame_kind")
                    if first_control_frame_kind:
                        stats["last_first_control_frame_kind"] = first_control_frame_kind
                    for field in (
                        "control_frame_only_timeout_ms",
                        "connect_first_frame_timeout_ms",
                        "seed_age_ms",
                        "venue_state_stale_ms",
                        "first_control_frame_seen",
                        "first_data_frame_seen",
                        "rest_seed_bridge_active",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=bootstrap_session_hedge" in line and "venue=extended" in line:
                    stats = extended_session_hedge_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    stats[f"action_{action}_count"] = (
                        safe_int(stats.get(f"action_{action}_count")) or 0
                    ) + 1
                    winner = pairs.get("winner")
                    if winner:
                        stats["last_winner"] = winner
                        stats[f"winner_{winner}_count"] = (
                            safe_int(stats.get(f"winner_{winner}_count")) or 0
                        ) + 1
                    loser = pairs.get("loser")
                    if loser:
                        stats["last_loser"] = loser
                    first_control_frame_kind = pairs.get("first_control_frame_kind")
                    if first_control_frame_kind:
                        stats["last_first_control_frame_kind"] = first_control_frame_kind
                    for field in (
                        "hedge_started_at_ms",
                        "connect_first_frame_timeout_ms",
                        "control_frame_only_timeout_ms",
                        "seed_age_ms",
                        "venue_state_stale_ms",
                        "rest_seed_bridge_active",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=backend_attach_fallback" in line and "venue=extended" in line:
                    stats = extended_backend_attach_fallback_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    stats[f"action_{action}_count"] = (
                        safe_int(stats.get(f"action_{action}_count")) or 0
                    ) + 1
                    for field in (
                        "winner_socket_role",
                        "winner_stream_kind",
                        "primary_stream_kind",
                        "fallback_stream_kind",
                    ):
                        value = pairs.get(field)
                        if value:
                            stats[f"last_{field}"] = value
                    for field in (
                        "hedge_started_at_ms",
                        "connect_first_frame_timeout_ms",
                        "control_frame_only_timeout_ms",
                        "seed_age_ms",
                        "rest_seed_bridge_active",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if (
                    "component=post_publish_stream_fallback" in line
                    and "venue=extended" in line
                ):
                    stats = extended_post_publish_fallback_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    stats[f"action_{action}_count"] = (
                        safe_int(stats.get(f"action_{action}_count")) or 0
                    ) + 1
                    for field in (
                        "active_stream_kind",
                        "fallback_stream_kind",
                        "winner_stream_kind",
                        "last_frame_kind",
                        "last_data_kind",
                        "stream_preference",
                    ):
                        value = pairs.get(field)
                        if value:
                            stats[f"last_{field}"] = value
                    for field in (
                        "started_at_ms",
                        "post_publish_fallback_after_ms",
                        "post_publish_fallback_deadline_ms",
                        "age_ws_rx_ms",
                        "age_data_rx_ms",
                        "age_book_event_ms",
                        "age_published_ms",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=bootstrap_timeout" in line and "venue=extended" in line:
                    reason = str(pairs.get("reason", "unknown")).lower()
                    stats = extended_bootstrap_timeout_stats[reason]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    stats["reason"] = reason
                    for field in (
                        "connect_first_frame_timeout_ms",
                        "connect_book_timeout_ms",
                        "rest_snapshot_seeded",
                        "rest_seed_bridge_active",
                        "first_control_frame_seen",
                        "first_data_frame_seen",
                        "first_message_seen",
                        "first_book_seen",
                        "first_publish_seen",
                        "stale_watchdog_armed",
                        "stale_watchdog_deferred_until_first_publish",
                        "rest_snapshot_seq",
                        "rest_snapshot_latency_ms",
                        "rest_snapshot_bid_levels",
                        "rest_snapshot_ask_levels",
                        "control_frame_only_timeout_ms",
                        "seed_age_ms",
                        "time_to_first_control_frame_ms",
                        "time_to_first_message_ms",
                        "time_to_first_book_ms",
                        "time_to_first_publish_ms",
                        "last_seq",
                        "last_snapshot_seq",
                        "last_book_seq",
                        "last_publish_seq",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for field in (
                        "last_frame_kind",
                        "last_data_kind",
                        "reason_family",
                        "first_control_frame_kind",
                    ):
                        value = pairs.get(field)
                        if value:
                            stats[f"last_{field}"] = value
                    timeout_stage = pairs.get("bootstrap_timeout_stage")
                    if timeout_stage:
                        stats["last_bootstrap_timeout_stage"] = timeout_stage
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)
                    extended_bootstrap_episodes.append(
                        {
                            "reason": reason,
                            "first_control_frame_seen": safe_int(
                                pairs.get("first_control_frame_seen")
                            )
                            or 0,
                            "first_data_frame_seen": safe_int(
                                pairs.get("first_data_frame_seen")
                            )
                            or 0,
                        }
                    )

                if "component=bootstrap_churn" in line and "venue=extended" in line:
                    stats = extended_bootstrap_churn_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    bootstrap_reason = pairs.get("bootstrap_reason")
                    if bootstrap_reason:
                        stats["last_bootstrap_reason"] = bootstrap_reason
                    for field in (
                        "bootstrap_count_window",
                        "bootstrap_window_ms",
                        "bootstrap_limit",
                        "bootstrap_fast_reconnect_allowed",
                        "bootstrap_churn_escalated",
                        "healthy_session_ms_before_reset",
                        "session_duration_ms",
                        "previous_bootstrap_count_window",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=stale_watchdog_churn" in line and "venue=extended" in line:
                    stats = extended_transport_gap_stats["extended"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    action = str(pairs.get("action", "unknown")).lower()
                    stats["last_action"] = action
                    for field in (
                        "stale_watchdog_count_window",
                        "stale_watchdog_window_ms",
                        "stale_watchdog_limit",
                        "stale_watchdog_fast_reconnect_allowed",
                        "stale_watchdog_churn_escalated",
                        "healthy_session_ms_before_reset",
                        "session_duration_ms",
                        "previous_stale_watchdog_count_window",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=session_progress" in line and "venue=extended" in line:
                    stats = extended_transport_gap_stats["extended"]
                    stats["session_progress_samples"] = (
                        safe_int(stats.get("session_progress_samples")) or 0
                    ) + 1
                    stage = str(pairs.get("stage", "unknown")).lower()
                    stats["last_stage"] = stage
                    socket_role = str(pairs.get("socket_role", "unknown")).lower()
                    role_stats = extended_socket_role_progress_stats[socket_role]
                    role_stats["samples"] = (
                        safe_int(role_stats.get("samples")) or 0
                    ) + 1
                    role_stats["last_stage"] = stage
                    role_stats[f"stage_{stage}_count"] = (
                        safe_int(role_stats.get(f"stage_{stage}_count")) or 0
                    ) + 1
                    stream_kind = str(pairs.get("stream_kind", "unknown")).lower()
                    stream_stats = extended_stream_kind_progress_stats[stream_kind]
                    stream_stats["samples"] = (
                        safe_int(stream_stats.get("samples")) or 0
                    ) + 1
                    stream_stats["last_stage"] = stage
                    stream_stats[f"stage_{stage}_count"] = (
                        safe_int(stream_stats.get(f"stage_{stage}_count")) or 0
                    ) + 1
                    ws_upgrade_completed = safe_int(pairs.get("ws_upgrade_completed"))
                    if ws_upgrade_completed is not None:
                        role_stats["last_ws_upgrade_completed"] = ws_upgrade_completed
                        update_max_stat(
                            role_stats,
                            "max_ws_upgrade_completed",
                            ws_upgrade_completed,
                        )
                        stream_stats["last_ws_upgrade_completed"] = ws_upgrade_completed
                        update_max_stat(
                            stream_stats,
                            "max_ws_upgrade_completed",
                            ws_upgrade_completed,
                        )
                    for field in (
                        "time_to_first_control_frame_ms",
                        "time_to_first_message_ms",
                        "time_to_first_book_ms",
                        "time_to_first_publish_ms",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)
                            role_stats[f"last_{field}"] = value
                            update_max_stat(role_stats, f"max_{field}", value)
                            stream_stats[f"last_{field}"] = value
                            update_max_stat(stream_stats, f"max_{field}", value)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)
                        update_max_numeric_token(role_stats, key, raw)
                        update_max_numeric_token(stream_stats, key, raw)

                if (
                    "component=watchdog_bootstrap_transition" in line
                    and "venue=extended" in line
                ):
                    stats = extended_transport_gap_stats["extended"]
                    stats["watchdog_bootstrap_transition_samples"] = (
                        safe_int(stats.get("watchdog_bootstrap_transition_samples")) or 0
                    ) + 1
                    for field in (
                        "first_publish_observed",
                        "watchdog_armed_now",
                        "stale_watchdog_deferred_until_first_publish",
                        "time_to_first_publish_ms",
                    ):
                        value = safe_int(pairs.get(field))
                        if value is not None:
                            stats[f"last_{field}"] = value
                            update_max_stat(stats, f"max_{field}", value)

                if "venue=lighter" in line and "lighter_ts_fallback_count=" in line:
                    stats = lighter_ts_fallback_stats["lighter"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    context = pairs.get("context")
                    if context:
                        stats["last_context"] = context
                    raw_ts = pairs.get("raw_ts")
                    if raw_ts:
                        stats["last_raw_ts"] = raw_ts
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "venue=lighter" in line and (
                    "lighter_ping_sent_count=" in line or "lighter_ping_send_fail_count=" in line
                ):
                    stats = ping_stats["lighter"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    err = pairs.get("err")
                    if err:
                        stats["last_err"] = err
                    sent = safe_int(pairs.get("lighter_ping_sent_count"))
                    fail = safe_int(pairs.get("lighter_ping_send_fail_count"))
                    update_max_stat(stats, "max_ping_sent_count", sent)
                    update_max_stat(stats, "max_ping_send_fail_count", fail)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "venue=paradex" in line and (
                    "paradex_ping_sent_count=" in line or "paradex_ping_send_fail_count=" in line
                ):
                    stats = ping_stats["paradex"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    err = pairs.get("err")
                    if err:
                        stats["last_err"] = err
                    sent = safe_int(pairs.get("paradex_ping_sent_count"))
                    fail = safe_int(pairs.get("paradex_ping_send_fail_count"))
                    update_max_stat(stats, "max_ping_sent_count", sent)
                    update_max_stat(stats, "max_ping_send_fail_count", fail)
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

                if "component=book_recovery" in line and "venue=aster" in line:
                    stage = str(pairs.get("stage", "unknown"))
                    phase = str(pairs.get("phase", "unknown"))
                    stats = aster_book_recovery_stats[f"{stage}/{phase}"]
                    stats["samples"] = (safe_int(stats.get("samples")) or 0) + 1
                    stats["stage"] = stage
                    stats["phase"] = phase
                    failure_class = pairs.get("failure_class")
                    if failure_class:
                        stats["last_failure_class"] = failure_class
                    for key, raw in pairs.items():
                        update_max_numeric_token(stats, key, raw)

            lower = line.lower()
            reason = infer_reason_from_line(lower)
            if reason is None:
                continue
            venue = infer_venue_from_line(lower) or "unknown"
            signature_reconnect_counts[(venue, reason)] += 1

    return (
        audit_reconnect_counts,
        signature_reconnect_counts,
        market_publisher_counters,
        runner_apply_stats,
        runner_apply_truth_stats,
        rest_monitor_stats,
        arb_gate_stats,
        hl_pubq_stats,
        extended_ws_msg_stats,
        extended_cfg_stats,
        extended_reconnect_policy_stats,
        extended_transport_gap_stats,
        summarize_extended_rest_seed(extended_rest_seed_stats),
        summarize_extended_seed_bridge(extended_seed_bridge_stats),
        summarize_extended_control_frame_grace(extended_control_frame_grace_stats),
        extended_session_hedge_stats,
        summarize_extended_backend_attach_fallback(
            extended_backend_attach_fallback_stats
        ),
        extended_bootstrap_timeout_stats,
        extended_bootstrap_churn_stats,
        summarize_extended_bootstrap(extended_bootstrap_episodes),
        ping_stats,
        lighter_ts_fallback_stats,
        aster_book_recovery_stats,
        summarize_extended_defects(extended_defect_episodes),
        summarize_extended_control_frame_before_data(extended_bootstrap_episodes),
        summarize_extended_socket_establishment(extended_socket_establishment_stats),
        summarize_extended_socket_role_progress(extended_socket_role_progress_stats),
        summarize_extended_stream_kind_progress(extended_stream_kind_progress_stats),
        summarize_extended_post_publish_fallback(extended_post_publish_fallback_stats),
        summarize_extended_post_publish_gap_stage(
            summarize_extended_post_publish_fallback(extended_post_publish_fallback_stats)
        ),
        summarize_extended_stream_preference(
            summarize_extended_post_publish_fallback(extended_post_publish_fallback_stats)
        ),
    )


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


def summarize_extended_seed_bridge(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    return {
        "samples": safe_int(summary.get("samples")) or 0,
        "activated_count": safe_int(summary.get("action_activated_count")) or 0,
        "cleared_count": safe_int(summary.get("action_cleared_count")) or 0,
        "last_action": summary.get("last_action"),
        "last_clear_reason": summary.get("last_clear_reason"),
        "last_rest_snapshot_seeded": safe_int(summary.get("last_rest_snapshot_seeded")),
        "last_rest_seed_bridge_active": safe_int(
            summary.get("last_rest_seed_bridge_active")
        ),
        "last_seed_age_ms": safe_int(summary.get("last_seed_age_ms")),
        "max_seed_age_ms": safe_int(summary.get("max_seed_age_ms")),
        "last_venue_state_stale_ms": safe_int(
            summary.get("last_venue_state_stale_ms")
        ),
        "last_connect_first_frame_timeout_ms": safe_int(
            summary.get("last_connect_first_frame_timeout_ms")
        ),
    }


def summarize_extended_control_frame_grace(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    return {
        "samples": safe_int(summary.get("samples")) or 0,
        "armed_count": safe_int(summary.get("action_armed_count")) or 0,
        "cleared_count": safe_int(summary.get("action_cleared_count")) or 0,
        "expired_count": safe_int(summary.get("action_expired_count")) or 0,
        "last_action": summary.get("last_action"),
        "last_reason_family": summary.get("last_reason_family"),
        "last_first_control_frame_kind": summary.get("last_first_control_frame_kind"),
        "last_first_control_frame_seen": safe_int(
            summary.get("last_first_control_frame_seen")
        ),
        "last_first_data_frame_seen": safe_int(
            summary.get("last_first_data_frame_seen")
        ),
        "last_rest_seed_bridge_active": safe_int(
            summary.get("last_rest_seed_bridge_active")
        ),
        "last_seed_age_ms": safe_int(summary.get("last_seed_age_ms")),
        "max_seed_age_ms": safe_int(summary.get("max_seed_age_ms")),
        "last_venue_state_stale_ms": safe_int(
            summary.get("last_venue_state_stale_ms")
        ),
        "last_connect_first_frame_timeout_ms": safe_int(
            summary.get("last_connect_first_frame_timeout_ms")
        ),
        "last_control_frame_only_timeout_ms": safe_int(
            summary.get("last_control_frame_only_timeout_ms")
        ),
    }


def summarize_extended_session_hedge(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    return {
        "samples": safe_int(summary.get("samples")) or 0,
        "started_count": safe_int(summary.get("action_started_count")) or 0,
        "primary_won_count": safe_int(summary.get("action_primary_won_count")) or 0,
        "hedge_won_count": safe_int(summary.get("action_hedge_won_count")) or 0,
        "cancelled_count": safe_int(summary.get("action_cancelled_count")) or 0,
        "expired_count": safe_int(summary.get("action_expired_count")) or 0,
        "last_action": summary.get("last_action"),
        "last_winner": summary.get("last_winner"),
        "last_loser": summary.get("last_loser"),
        "last_first_control_frame_kind": summary.get("last_first_control_frame_kind"),
        "last_hedge_started_at_ms": safe_int(summary.get("last_hedge_started_at_ms")),
        "max_hedge_started_at_ms": safe_int(summary.get("max_hedge_started_at_ms")),
        "last_connect_first_frame_timeout_ms": safe_int(
            summary.get("last_connect_first_frame_timeout_ms")
        ),
        "last_control_frame_only_timeout_ms": safe_int(
            summary.get("last_control_frame_only_timeout_ms")
        ),
        "last_seed_age_ms": safe_int(summary.get("last_seed_age_ms")),
        "max_seed_age_ms": safe_int(summary.get("max_seed_age_ms")),
        "last_venue_state_stale_ms": safe_int(
            summary.get("last_venue_state_stale_ms")
        ),
        "last_rest_seed_bridge_active": safe_int(
            summary.get("last_rest_seed_bridge_active")
        ),
    }


def summarize_extended_backend_attach_fallback(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    return {
        "samples": safe_int(summary.get("samples")) or 0,
        "started_count": safe_int(summary.get("action_started_count")) or 0,
        "primary_won_count": safe_int(summary.get("action_primary_won_count")) or 0,
        "fallback_won_count": safe_int(summary.get("action_fallback_won_count")) or 0,
        "cancelled_count": safe_int(summary.get("action_cancelled_count")) or 0,
        "expired_count": safe_int(summary.get("action_expired_count")) or 0,
        "last_action": summary.get("last_action"),
        "last_winner_socket_role": summary.get("last_winner_socket_role"),
        "last_winner_stream_kind": summary.get("last_winner_stream_kind"),
        "last_primary_stream_kind": summary.get("last_primary_stream_kind"),
        "last_fallback_stream_kind": summary.get("last_fallback_stream_kind"),
        "last_hedge_started_at_ms": safe_int(summary.get("last_hedge_started_at_ms")),
        "max_hedge_started_at_ms": safe_int(summary.get("max_hedge_started_at_ms")),
        "last_connect_first_frame_timeout_ms": safe_int(
            summary.get("last_connect_first_frame_timeout_ms")
        ),
        "last_control_frame_only_timeout_ms": safe_int(
            summary.get("last_control_frame_only_timeout_ms")
        ),
        "last_seed_age_ms": safe_int(summary.get("last_seed_age_ms")),
        "max_seed_age_ms": safe_int(summary.get("max_seed_age_ms")),
        "last_rest_seed_bridge_active": safe_int(
            summary.get("last_rest_seed_bridge_active")
        ),
    }


def summarize_extended_post_publish_fallback(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    return {
        "samples": safe_int(summary.get("samples")) or 0,
        "armed_count": safe_int(summary.get("action_armed_count")) or 0,
        "started_count": safe_int(summary.get("action_started_count")) or 0,
        "primary_recovered_count": safe_int(
            summary.get("action_primary_recovered_count")
        )
        or 0,
        "fallback_won_count": safe_int(summary.get("action_fallback_won_count")) or 0,
        "expired_count": safe_int(summary.get("action_expired_count")) or 0,
        "cancelled_count": safe_int(summary.get("action_cancelled_count")) or 0,
        "preference_set_count": safe_int(summary.get("action_preference_set_count"))
        or 0,
        "preference_reset_count": safe_int(
            summary.get("action_preference_reset_count")
        )
        or 0,
        "last_action": summary.get("last_action"),
        "last_active_stream_kind": summary.get("last_active_stream_kind"),
        "last_fallback_stream_kind": summary.get("last_fallback_stream_kind"),
        "last_winner_stream_kind": summary.get("last_winner_stream_kind"),
        "last_last_frame_kind": summary.get("last_last_frame_kind"),
        "last_last_data_kind": summary.get("last_last_data_kind"),
        "last_stream_preference": summary.get("last_stream_preference"),
        "last_started_at_ms": safe_int(summary.get("last_started_at_ms")),
        "max_started_at_ms": safe_int(summary.get("max_started_at_ms")),
        "last_post_publish_fallback_after_ms": safe_int(
            summary.get("last_post_publish_fallback_after_ms")
        ),
        "last_post_publish_fallback_deadline_ms": safe_int(
            summary.get("last_post_publish_fallback_deadline_ms")
        ),
        "last_age_ws_rx_ms": safe_int(summary.get("last_age_ws_rx_ms")),
        "last_age_data_rx_ms": safe_int(summary.get("last_age_data_rx_ms")),
        "last_age_book_event_ms": safe_int(summary.get("last_age_book_event_ms")),
        "last_age_published_ms": safe_int(summary.get("last_age_published_ms")),
        "max_age_ws_rx_ms": safe_int(summary.get("max_age_ws_rx_ms")),
        "max_age_data_rx_ms": safe_int(summary.get("max_age_data_rx_ms")),
        "max_age_book_event_ms": safe_int(summary.get("max_age_book_event_ms")),
        "max_age_published_ms": safe_int(summary.get("max_age_published_ms")),
    }


def summarize_extended_post_publish_gap_stage(
    summary: dict[str, Any]
) -> dict[str, Any]:
    counts = {
        "armed": safe_int(summary.get("armed_count")) or 0,
        "started": safe_int(summary.get("started_count")) or 0,
        "primary_recovered": safe_int(summary.get("primary_recovered_count")) or 0,
        "fallback_won": safe_int(summary.get("fallback_won_count")) or 0,
        "expired": safe_int(summary.get("expired_count")) or 0,
        "cancelled": safe_int(summary.get("cancelled_count")) or 0,
    }
    terminal_counts = {
        "primary_recovered": counts["primary_recovered"],
        "fallback_won": counts["fallback_won"],
        "expired": counts["expired"],
        "cancelled": counts["cancelled"],
    }
    attempts = counts["started"]
    dominant_stage = "none"
    dominant_count = 0
    if attempts > 0:
        dominant_stage, dominant_count = max(
            terminal_counts.items(), key=lambda item: item[1]
        )
    confidence_pct = (dominant_count / attempts * 100.0) if attempts > 0 else 0.0
    return {
        "counts": counts,
        "attempts": attempts,
        "dominant_stage": dominant_stage,
        "dominant_count": dominant_count,
        "confidence_pct": confidence_pct,
        "successful_recoveries": counts["primary_recovered"] + counts["fallback_won"],
    }


def summarize_extended_stream_preference(summary: dict[str, Any]) -> dict[str, Any]:
    last_preference = str(summary.get("last_stream_preference") or "depth1")
    return {
        "last_stream_preference": last_preference,
        "preference_set_count": safe_int(summary.get("preference_set_count")) or 0,
        "preference_reset_count": safe_int(summary.get("preference_reset_count")) or 0,
        "degraded_active": 1 if last_preference == "full_orderbook_degraded" else 0,
    }


def summarize_extended_control_frame_before_data(
    episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = {
        "no_frame": 0,
        "control_frame_only": 0,
        "data_frame_seen": 0,
    }
    for episode in episodes:
        first_data = safe_int(episode.get("first_data_frame_seen")) or 0
        first_control = safe_int(episode.get("first_control_frame_seen")) or 0
        if first_data:
            counts["data_frame_seen"] += 1
        elif first_control:
            counts["control_frame_only"] += 1
        else:
            counts["no_frame"] += 1
    total = sum(counts.values())
    dominant_shape = "no_frame"
    dominant_count = 0
    if total > 0:
        dominant_shape, dominant_count = max(counts.items(), key=lambda item: item[1])
    confidence_pct = (dominant_count / total * 100.0) if total > 0 else 0.0
    return {
        "counts": counts,
        "total": total,
        "dominant_shape": dominant_shape,
        "dominant_count": dominant_count,
        "confidence_pct": confidence_pct,
    }


def summarize_extended_socket_establishment(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary = dict(stats.get("extended", {}))
    return {
        "samples": safe_int(summary.get("samples")) or 0,
        "tcp_connected_count": safe_int(summary.get("action_tcp_connected_count")) or 0,
        "ws_upgraded_count": safe_int(summary.get("action_ws_upgraded_count")) or 0,
        "failed_count": safe_int(summary.get("action_failed_count")) or 0,
        "primary_ws_upgraded_count": safe_int(summary.get("primary_ws_upgraded_count")) or 0,
        "hedge_ws_upgraded_count": safe_int(summary.get("hedge_ws_upgraded_count")) or 0,
        "primary_failed_count": safe_int(summary.get("primary_failed_count")) or 0,
        "hedge_failed_count": safe_int(summary.get("hedge_failed_count")) or 0,
        "last_action": summary.get("last_action"),
        "last_socket_role": summary.get("last_socket_role"),
        "last_stream_kind": summary.get("last_stream_kind"),
        "last_host": summary.get("last_host"),
        "last_path": summary.get("last_path"),
        "last_failure_stage": summary.get("last_failure_stage"),
        "last_failure_class": summary.get("last_failure_class"),
        "last_disable_nagle": safe_int(summary.get("last_disable_nagle")),
        "last_elapsed_ms": safe_int(summary.get("last_elapsed_ms")),
        "max_elapsed_ms": safe_int(summary.get("max_elapsed_ms")),
        "last_tcp_connect_ms": safe_int(summary.get("last_tcp_connect_ms")),
        "max_tcp_connect_ms": safe_int(summary.get("max_tcp_connect_ms")),
        "last_ws_upgrade_ms": safe_int(summary.get("last_ws_upgrade_ms")),
        "max_ws_upgrade_ms": safe_int(summary.get("max_ws_upgrade_ms")),
        "depth1_ws_upgraded_count": safe_int(summary.get("depth1_ws_upgraded_count")) or 0,
        "full_orderbook_ws_upgraded_count": safe_int(
            summary.get("full_orderbook_ws_upgraded_count")
        )
        or 0,
    }


def summarize_extended_socket_role_progress(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for socket_role in ("primary", "hedge"):
        role = dict(stats.get(socket_role, {}))
        summary[socket_role] = {
            "samples": safe_int(role.get("samples")) or 0,
            "last_stage": role.get("last_stage"),
            "stage_first_control_frame_count": safe_int(
                role.get("stage_first_control_frame_count")
            )
            or 0,
            "stage_first_message_count": safe_int(
                role.get("stage_first_message_count")
            )
            or 0,
            "stage_first_book_count": safe_int(role.get("stage_first_book_count")) or 0,
            "stage_first_publish_count": safe_int(
                role.get("stage_first_publish_count")
            )
            or 0,
            "last_ws_upgrade_completed": safe_int(
                role.get("last_ws_upgrade_completed")
            ),
            "last_time_to_first_control_frame_ms": safe_int(
                role.get("last_time_to_first_control_frame_ms")
            ),
            "max_time_to_first_control_frame_ms": safe_int(
                role.get("max_time_to_first_control_frame_ms")
            ),
            "last_time_to_first_message_ms": safe_int(
                role.get("last_time_to_first_message_ms")
            ),
            "max_time_to_first_message_ms": safe_int(
                role.get("max_time_to_first_message_ms")
            ),
            "last_time_to_first_book_ms": safe_int(
                role.get("last_time_to_first_book_ms")
            ),
            "max_time_to_first_book_ms": safe_int(
                role.get("max_time_to_first_book_ms")
            ),
            "last_time_to_first_publish_ms": safe_int(
                role.get("last_time_to_first_publish_ms")
            ),
            "max_time_to_first_publish_ms": safe_int(
                role.get("max_time_to_first_publish_ms")
            ),
        }
    return summary


def summarize_extended_stream_kind_progress(
    stats: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stream_kind in ("depth1", "full_orderbook"):
        stream = dict(stats.get(stream_kind, {}))
        summary[stream_kind] = {
            "samples": safe_int(stream.get("samples")) or 0,
            "last_stage": stream.get("last_stage"),
            "stage_first_control_frame_count": safe_int(
                stream.get("stage_first_control_frame_count")
            )
            or 0,
            "stage_first_message_count": safe_int(
                stream.get("stage_first_message_count")
            )
            or 0,
            "stage_first_book_count": safe_int(stream.get("stage_first_book_count"))
            or 0,
            "stage_first_publish_count": safe_int(
                stream.get("stage_first_publish_count")
            )
            or 0,
            "last_ws_upgrade_completed": safe_int(
                stream.get("last_ws_upgrade_completed")
            ),
            "last_time_to_first_control_frame_ms": safe_int(
                stream.get("last_time_to_first_control_frame_ms")
            ),
            "max_time_to_first_control_frame_ms": safe_int(
                stream.get("max_time_to_first_control_frame_ms")
            ),
            "last_time_to_first_message_ms": safe_int(
                stream.get("last_time_to_first_message_ms")
            ),
            "max_time_to_first_message_ms": safe_int(
                stream.get("max_time_to_first_message_ms")
            ),
            "last_time_to_first_book_ms": safe_int(
                stream.get("last_time_to_first_book_ms")
            ),
            "max_time_to_first_book_ms": safe_int(
                stream.get("max_time_to_first_book_ms")
            ),
            "last_time_to_first_publish_ms": safe_int(
                stream.get("last_time_to_first_publish_ms")
            ),
            "max_time_to_first_publish_ms": safe_int(
                stream.get("max_time_to_first_publish_ms")
            ),
        }
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
        value = market_publisher_counters.get(name, 0)
        if value != 0:
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
    runner_apply_stats: dict[str, dict[str, Any]],
    runner_apply_truth_stats: dict[str, dict[str, Any]],
    rest_monitor_stats: dict[str, dict[str, Any]],
    arb_gate_stats: dict[str, dict[str, Any]],
    hl_pubq_stats: dict[str, dict[str, Any]],
    extended_ws_msg_stats: dict[str, dict[str, Any]],
    extended_cfg_stats: dict[str, dict[str, Any]],
    extended_reconnect_policy_stats: dict[str, dict[str, Any]],
    extended_transport_gap_stats: dict[str, dict[str, Any]],
    extended_rest_seed_summary: dict[str, Any],
    extended_seed_bridge_summary: dict[str, Any],
    extended_control_frame_grace_summary: dict[str, Any],
    extended_session_hedge_summary: dict[str, Any],
    extended_backend_attach_fallback_summary: dict[str, Any],
    extended_post_publish_fallback_summary: dict[str, Any],
    extended_post_publish_gap_stage_summary: dict[str, Any],
    extended_stream_preference_summary: dict[str, Any],
    extended_bootstrap_timeout_stats: dict[str, dict[str, Any]],
    extended_bootstrap_churn_stats: dict[str, dict[str, Any]],
    extended_bootstrap_summary: dict[str, Any],
    extended_control_frame_before_data_summary: dict[str, Any],
    extended_socket_establishment_summary: dict[str, Any],
    extended_socket_role_progress_summary: dict[str, Any],
    extended_stream_kind_progress_summary: dict[str, Any],
    ping_stats: dict[str, dict[str, Any]],
    lighter_ts_fallback_stats: dict[str, dict[str, Any]],
    aster_book_recovery_stats: dict[str, dict[str, Any]],
    extended_defect_summary: dict[str, Any],
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

    lines.append("## Extended REST Snapshot Seed")
    rest_seed_stats = dict(extended_rest_seed_summary.get("extended", {}))
    if rest_seed_stats:
        lines.append(
            md_table(
                [
                    "samples",
                    "status_ok_count",
                    "status_http_error_count",
                    "status_parse_error_count",
                    "status_empty_count",
                    "last_http_status",
                    "last_latency_ms",
                    "max_latency_ms",
                    "last_seeded",
                    "last_bid_levels",
                    "last_ask_levels",
                ],
                [[
                    fmt_int(safe_int(rest_seed_stats.get("samples"))),
                    fmt_int(safe_int(rest_seed_stats.get("status_ok_count"))),
                    fmt_int(safe_int(rest_seed_stats.get("status_http_error_count"))),
                    fmt_int(safe_int(rest_seed_stats.get("status_parse_error_count"))),
                    fmt_int(safe_int(rest_seed_stats.get("status_empty_count"))),
                    fmt_int(safe_int(rest_seed_stats.get("last_http_status"))),
                    fmt_int(safe_int(rest_seed_stats.get("last_latency_ms"))),
                    fmt_int(safe_int(rest_seed_stats.get("max_latency_ms"))),
                    fmt_int(safe_int(rest_seed_stats.get("last_seeded"))),
                    fmt_int(safe_int(rest_seed_stats.get("last_bid_levels"))),
                    fmt_int(safe_int(rest_seed_stats.get("last_ask_levels"))),
                ]],
            )
        )
        failures_by_status = extended_rest_seed_summary.get("failures_by_status") or {}
        failures_by_http_status = extended_rest_seed_summary.get("failures_by_http_status") or {}
        if failures_by_status:
            lines.append(
                f"- failure counts by status: `{json.dumps(failures_by_status, sort_keys=True)}`"
            )
        if failures_by_http_status:
            lines.append(
                f"- failure counts by HTTP status: `{json.dumps(failures_by_http_status, sort_keys=True)}`"
            )
    else:
        lines.append("_No `component=rest_snapshot_seed` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Seed Bridge")
    if safe_int(extended_seed_bridge_summary.get("samples")):
        lines.append(
            md_table(
                [
                    "samples",
                    "activated_count",
                    "cleared_count",
                    "last_action",
                    "last_clear_reason",
                    "last_seed_age_ms",
                    "max_seed_age_ms",
                    "last_rest_seed_bridge_active",
                    "last_venue_state_stale_ms",
                    "last_connect_first_frame_timeout_ms",
                ],
                [[
                    fmt_int(safe_int(extended_seed_bridge_summary.get("samples"))),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("activated_count"))),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("cleared_count"))),
                    str(extended_seed_bridge_summary.get("last_action") or "n/a"),
                    str(extended_seed_bridge_summary.get("last_clear_reason") or "n/a"),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("last_seed_age_ms"))),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("max_seed_age_ms"))),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("last_rest_seed_bridge_active"))),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("last_venue_state_stale_ms"))),
                    fmt_int(safe_int(extended_seed_bridge_summary.get("last_connect_first_frame_timeout_ms"))),
                ]],
            )
        )
    else:
        lines.append("_No `component=bootstrap_seed_bridge` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Control-Frame Grace")
    if safe_int(extended_control_frame_grace_summary.get("samples")):
        lines.append(
            "- armed / cleared / expired: "
            f"`{fmt_int(safe_int(extended_control_frame_grace_summary.get('armed_count')))} / "
            f"{fmt_int(safe_int(extended_control_frame_grace_summary.get('cleared_count')))} / "
            f"{fmt_int(safe_int(extended_control_frame_grace_summary.get('expired_count')))}`"
        )
        lines.append(
            "- last action / reason_family: "
            f"`{extended_control_frame_grace_summary.get('last_action', 'n/a')} / "
            f"{extended_control_frame_grace_summary.get('last_reason_family', 'n/a')}`"
        )
        lines.append(
            "- last first-control kind / first-data-seen / bridge-active: "
            f"`{extended_control_frame_grace_summary.get('last_first_control_frame_kind', 'n/a')} / "
            f"{fmt_int(safe_int(extended_control_frame_grace_summary.get('last_first_data_frame_seen')))} / "
            f"{fmt_int(safe_int(extended_control_frame_grace_summary.get('last_rest_seed_bridge_active')))} `"
        )
        lines.append(
            "- last control-frame-only timeout / seed-age / state-stale: "
            f"`{fmt_int(safe_int(extended_control_frame_grace_summary.get('last_control_frame_only_timeout_ms')))}ms / "
            f"{fmt_int(safe_int(extended_control_frame_grace_summary.get('last_seed_age_ms')))}ms / "
            f"{fmt_int(safe_int(extended_control_frame_grace_summary.get('last_venue_state_stale_ms')))}ms`"
        )
    else:
        lines.append("_No `component=bootstrap_control_frame_grace` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Session Hedge")
    if safe_int(extended_session_hedge_summary.get("samples")):
        lines.append(
            "- started / primary_won / hedge_won / cancelled / expired: "
            f"`{fmt_int(safe_int(extended_session_hedge_summary.get('started_count')))} / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('primary_won_count')))} / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('hedge_won_count')))} / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('cancelled_count')))} / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('expired_count')))}`"
        )
        lines.append(
            "- last action / winner / loser: "
            f"`{extended_session_hedge_summary.get('last_action', 'n/a')} / "
            f"{extended_session_hedge_summary.get('last_winner', 'n/a')} / "
            f"{extended_session_hedge_summary.get('last_loser', 'n/a')}`"
        )
        lines.append(
            "- last hedge-start / first-control kind / bridge-active: "
            f"`{fmt_int(safe_int(extended_session_hedge_summary.get('last_hedge_started_at_ms')))}ms / "
            f"{extended_session_hedge_summary.get('last_first_control_frame_kind', 'n/a')} / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('last_rest_seed_bridge_active')))}`"
        )
        lines.append(
            "- last control-frame-only timeout / seed-age / state-stale: "
            f"`{fmt_int(safe_int(extended_session_hedge_summary.get('last_control_frame_only_timeout_ms')))}ms / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('last_seed_age_ms')))}ms / "
            f"{fmt_int(safe_int(extended_session_hedge_summary.get('last_venue_state_stale_ms')))}ms`"
        )
    else:
        lines.append("_No `component=bootstrap_session_hedge` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Backend-Attach Fallback")
    if safe_int(extended_backend_attach_fallback_summary.get("samples")):
        lines.append(
            "- started / primary_won / fallback_won / cancelled / expired: "
            f"`{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('started_count')))} / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('primary_won_count')))} / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('fallback_won_count')))} / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('cancelled_count')))} / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('expired_count')))}`"
        )
        lines.append(
            "- last action / winner role / winner stream: "
            f"`{extended_backend_attach_fallback_summary.get('last_action', 'n/a')} / "
            f"{extended_backend_attach_fallback_summary.get('last_winner_socket_role', 'n/a')} / "
            f"{extended_backend_attach_fallback_summary.get('last_winner_stream_kind', 'n/a')}`"
        )
        lines.append(
            "- last primary stream / fallback stream / bridge-active: "
            f"`{extended_backend_attach_fallback_summary.get('last_primary_stream_kind', 'n/a')} / "
            f"{extended_backend_attach_fallback_summary.get('last_fallback_stream_kind', 'n/a')} / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('last_rest_seed_bridge_active')))}`"
        )
        lines.append(
            "- last start / timeout / seed-age: "
            f"`{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('last_hedge_started_at_ms')))}ms / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('last_control_frame_only_timeout_ms')))}ms / "
            f"{fmt_int(safe_int(extended_backend_attach_fallback_summary.get('last_seed_age_ms')))}ms`"
        )
    else:
        lines.append("_No `component=backend_attach_fallback` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Post-Publish Stream Fallback")
    if safe_int(extended_post_publish_fallback_summary.get("samples")):
        lines.append(
            "- armed / started / primary_recovered / fallback_won / expired / cancelled: "
            f"`{fmt_int(safe_int(extended_post_publish_fallback_summary.get('armed_count')))} / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('started_count')))} / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('primary_recovered_count')))} / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('fallback_won_count')))} / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('expired_count')))} / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('cancelled_count')))}`"
        )
        lines.append(
            "- dominant post-publish stage / confidence: "
            f"`{extended_post_publish_gap_stage_summary.get('dominant_stage', 'none')} / "
            f"{safe_float(extended_post_publish_gap_stage_summary.get('confidence_pct')) or 0.0:.1f}%`"
        )
        lines.append(
            "- last action / active stream / winner stream / preference: "
            f"`{extended_post_publish_fallback_summary.get('last_action', 'n/a')} / "
            f"{extended_post_publish_fallback_summary.get('last_active_stream_kind', 'n/a')} / "
            f"{extended_post_publish_fallback_summary.get('last_winner_stream_kind', 'n/a')} / "
            f"{extended_stream_preference_summary.get('last_stream_preference', 'depth1')}`"
        )
        lines.append(
            "- last after / deadline / data age / published age: "
            f"`{fmt_int(safe_int(extended_post_publish_fallback_summary.get('last_post_publish_fallback_after_ms')))}ms / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('last_post_publish_fallback_deadline_ms')))}ms / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('last_age_data_rx_ms')))}ms / "
            f"{fmt_int(safe_int(extended_post_publish_fallback_summary.get('last_age_published_ms')))}ms`"
        )
    else:
        lines.append("_No `component=post_publish_stream_fallback` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Socket Establishment")
    if safe_int(extended_socket_establishment_summary.get("samples")):
        lines.append(
            "- tcp_connected / ws_upgraded / failed: "
            f"`{fmt_int(safe_int(extended_socket_establishment_summary.get('tcp_connected_count')))} / "
            f"{fmt_int(safe_int(extended_socket_establishment_summary.get('ws_upgraded_count')))} / "
            f"{fmt_int(safe_int(extended_socket_establishment_summary.get('failed_count')))}`"
        )
        lines.append(
            "- primary ws_upgraded / hedge ws_upgraded: "
            f"`{fmt_int(safe_int(extended_socket_establishment_summary.get('primary_ws_upgraded_count')))} / "
            f"{fmt_int(safe_int(extended_socket_establishment_summary.get('hedge_ws_upgraded_count')))}`"
        )
        lines.append(
            "- depth1 ws_upgraded / full_orderbook ws_upgraded: "
            f"`{fmt_int(safe_int(extended_socket_establishment_summary.get('depth1_ws_upgraded_count')))} / "
            f"{fmt_int(safe_int(extended_socket_establishment_summary.get('full_orderbook_ws_upgraded_count')))}`"
        )
        lines.append(
            "- last action / role / stream / failure stage / failure class: "
            f"`{extended_socket_establishment_summary.get('last_action', 'n/a')} / "
            f"{extended_socket_establishment_summary.get('last_socket_role', 'n/a')} / "
            f"{extended_socket_establishment_summary.get('last_stream_kind', 'n/a')} / "
            f"{extended_socket_establishment_summary.get('last_failure_stage', 'n/a')} / "
            f"{extended_socket_establishment_summary.get('last_failure_class', 'n/a')}`"
        )
        lines.append(
            "- last tcp_connect / ws_upgrade / elapsed: "
            f"`{fmt_int(safe_int(extended_socket_establishment_summary.get('last_tcp_connect_ms')))}ms / "
            f"{fmt_int(safe_int(extended_socket_establishment_summary.get('last_ws_upgrade_ms')))}ms / "
            f"{fmt_int(safe_int(extended_socket_establishment_summary.get('last_elapsed_ms')))}ms`"
        )
        primary_role = extended_socket_role_progress_summary.get("primary", {})
        hedge_role = extended_socket_role_progress_summary.get("hedge", {})
        lines.append(
            "- primary last stage / max first-message / max first-publish: "
            f"`{primary_role.get('last_stage', 'n/a')} / "
            f"{fmt_int(safe_int(primary_role.get('max_time_to_first_message_ms')))}ms / "
            f"{fmt_int(safe_int(primary_role.get('max_time_to_first_publish_ms')))}ms`"
        )
        lines.append(
            "- hedge last stage / max first-message / max first-publish: "
            f"`{hedge_role.get('last_stage', 'n/a')} / "
            f"{fmt_int(safe_int(hedge_role.get('max_time_to_first_message_ms')))}ms / "
            f"{fmt_int(safe_int(hedge_role.get('max_time_to_first_publish_ms')))}ms`"
        )
        depth1_stream = extended_stream_kind_progress_summary.get("depth1", {})
        full_stream = extended_stream_kind_progress_summary.get("full_orderbook", {})
        lines.append(
            "- depth1 last stage / max first-message / max first-publish: "
            f"`{depth1_stream.get('last_stage', 'n/a')} / "
            f"{fmt_int(safe_int(depth1_stream.get('max_time_to_first_message_ms')))}ms / "
            f"{fmt_int(safe_int(depth1_stream.get('max_time_to_first_publish_ms')))}ms`"
        )
        lines.append(
            "- full_orderbook last stage / max first-message / max first-publish: "
            f"`{full_stream.get('last_stage', 'n/a')} / "
            f"{fmt_int(safe_int(full_stream.get('max_time_to_first_message_ms')))}ms / "
            f"{fmt_int(safe_int(full_stream.get('max_time_to_first_publish_ms')))}ms`"
        )
    else:
        lines.append("_No `component=socket_establishment` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Bootstrap Truth")
    lines.append(
        f"- dominant bootstrap subtype: `{extended_bootstrap_summary.get('dominant_reason', 'bootstrap_no_first_frame')}`"
    )
    lines.append(
        f"- dominant count / total: `{fmt_int(safe_int(extended_bootstrap_summary.get('dominant_count')))} / {fmt_int(safe_int(extended_bootstrap_summary.get('total')))} ({safe_float(extended_bootstrap_summary.get('confidence_pct')) or 0.0:.2f}%)`"
    )
    bootstrap_stage_summary = summarize_extended_bootstrap_stages(
        extended_bootstrap_timeout_stats
    )
    lines.append(
        f"- dominant bootstrap timeout stage: `{bootstrap_stage_summary.get('dominant_stage', 'first_frame')}`"
    )
    lines.append(
        f"- dominant pre-first-data shape: `{extended_control_frame_before_data_summary.get('dominant_shape', 'no_frame')}`"
    )
    lines.append("")

    if extended_bootstrap_timeout_stats:
        rows = []
        for reason, stats in sorted(extended_bootstrap_timeout_stats.items()):
            rows.append(
                [
                    reason,
                    fmt_int(safe_int(stats.get("samples"))),
                    str(stats.get("last_bootstrap_timeout_stage") or "n/a"),
                    fmt_int(safe_int(stats.get("last_connect_first_frame_timeout_ms"))),
                    fmt_int(safe_int(stats.get("last_connect_book_timeout_ms"))),
                    fmt_int(safe_int(stats.get("last_rest_snapshot_seeded"))),
                    fmt_int(safe_int(stats.get("last_rest_seed_bridge_active"))),
                    fmt_int(safe_int(stats.get("last_first_control_frame_seen"))),
                    str(stats.get("last_first_control_frame_kind") or "n/a"),
                    fmt_int(safe_int(stats.get("last_first_data_frame_seen"))),
                    fmt_int(safe_int(stats.get("last_first_message_seen"))),
                    fmt_int(safe_int(stats.get("last_first_book_seen"))),
                    fmt_int(safe_int(stats.get("last_first_publish_seen"))),
                    fmt_int(
                        safe_int(
                            stats.get(
                                "last_stale_watchdog_deferred_until_first_publish"
                            )
                        )
                    ),
                    fmt_int(safe_int(stats.get("max_time_to_first_control_frame_ms"))),
                    fmt_int(safe_int(stats.get("max_time_to_first_message_ms"))),
                    fmt_int(safe_int(stats.get("max_time_to_first_book_ms"))),
                    fmt_int(safe_int(stats.get("max_time_to_first_publish_ms"))),
                    str(stats.get("last_last_frame_kind") or "n/a"),
                    str(stats.get("last_last_data_kind") or "n/a"),
                ]
            )
        lines.append(
            md_table(
                [
                    "reason",
                    "samples",
                    "last_bootstrap_timeout_stage",
                    "last_connect_first_frame_timeout_ms",
                    "last_connect_book_timeout_ms",
                    "last_rest_snapshot_seeded",
                    "last_rest_seed_bridge_active",
                    "last_first_control_frame_seen",
                    "last_first_control_frame_kind",
                    "last_first_data_frame_seen",
                    "last_first_message_seen",
                    "last_first_book_seen",
                    "last_first_publish_seen",
                    "last_watchdog_deferred",
                    "max_time_to_first_control_frame_ms",
                    "max_time_to_first_message_ms",
                    "max_time_to_first_book_ms",
                    "max_time_to_first_publish_ms",
                    "last_frame_kind",
                    "last_data_kind",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=bootstrap_timeout` entries found in run.log._")
    lines.append("")

    if extended_bootstrap_churn_stats:
        rows = []
        for venue, stats in sorted(extended_bootstrap_churn_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    str(stats.get("last_action") or "n/a"),
                    str(stats.get("last_bootstrap_reason") or "n/a"),
                    fmt_int(safe_int(stats.get("max_bootstrap_count_window"))),
                    fmt_int(safe_int(stats.get("last_bootstrap_limit"))),
                    fmt_int(safe_int(stats.get("last_bootstrap_window_ms"))),
                    fmt_int(safe_int(stats.get("last_bootstrap_fast_reconnect_allowed"))),
                    fmt_int(safe_int(stats.get("last_bootstrap_churn_escalated"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "last_action",
                    "last_bootstrap_reason",
                    "max_bootstrap_count_window",
                    "last_bootstrap_limit",
                    "last_bootstrap_window_ms",
                    "last_fast_reconnect_allowed",
                    "last_bootstrap_churn_escalated",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=bootstrap_churn` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Reconnect Policy Audit (WS_AUDIT)")
    if extended_reconnect_policy_stats:
        rows = []
        for reason, stats in sorted(extended_reconnect_policy_stats.items()):
            rows.append(
                [
                    reason,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("last_sleep_ms"))),
                    fmt_int(safe_int(stats.get("max_sleep_ms"))),
                    fmt_int(
                        safe_int(stats.get("last_failure_escalation_suppressed"))
                    ),
                    fmt_int(safe_int(stats.get("last_consecutive_failures"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "reason",
                    "samples",
                    "last_sleep_ms",
                    "max_sleep_ms",
                    "last_failure_escalation_suppressed",
                    "last_consecutive_failures",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=reconnect_policy` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Transport-Gap Audit (WS_AUDIT)")
    if extended_transport_gap_stats:
        rows = []
        for venue, stats in sorted(extended_transport_gap_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(
                        safe_int(stats.get("watchdog_bootstrap_transition_samples"))
                    ),
                    fmt_int(safe_int(stats.get("max_stale_watchdog_count_window"))),
                    fmt_int(safe_int(stats.get("last_stale_watchdog_limit"))),
                    fmt_int(safe_int(stats.get("last_stale_watchdog_window_ms"))),
                    fmt_int(
                        safe_int(stats.get("last_stale_watchdog_fast_reconnect_allowed"))
                    ),
                    fmt_int(safe_int(stats.get("last_stale_watchdog_churn_escalated"))),
                    fmt_int(safe_int(stats.get("last_watchdog_armed_now"))),
                    fmt_int(
                        safe_int(
                            stats.get("last_stale_watchdog_deferred_until_first_publish")
                        )
                    ),
                    fmt_int(safe_int(stats.get("max_time_to_first_message_ms"))),
                    fmt_int(safe_int(stats.get("max_time_to_first_book_ms"))),
                    fmt_int(safe_int(stats.get("max_time_to_first_publish_ms"))),
                    fmt_int(safe_int(stats.get("max_healthy_session_ms_before_reset"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "churn_samples",
                    "watchdog_transition_samples",
                    "max_stale_watchdog_count_window",
                    "last_stale_watchdog_limit",
                    "last_stale_watchdog_window_ms",
                    "last_fast_reconnect_allowed",
                    "last_churn_escalated",
                    "last_watchdog_armed_now",
                    "last_watchdog_deferred",
                    "max_time_to_first_message_ms",
                    "max_time_to_first_book_ms",
                    "max_time_to_first_publish_ms",
                    "max_healthy_session_ms_before_reset",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=stale_watchdog_churn` or `component=session_progress` entries found in run.log._")
    lines.append("")

    lines.append("## MarketPublisher Pressure Counters (WS_AUDIT)")
    if market_publisher_counters:
        rows = [[name, str(value)] for name, value in sorted(market_publisher_counters.items())]
        lines.append(md_table(["counter", "max_count"], rows))
    else:
        lines.append("_No `component=market_publisher` counters found in run.log._")
    lines.append("")

    lines.append("## Runner Apply Audit (WS_AUDIT)")
    if runner_apply_stats:
        rows: list[list[str]] = []
        for venue, stats in sorted(runner_apply_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_cache_err"))),
                    fmt_int(safe_int(stats.get("max_ext_future"))),
                    fmt_ms(safe_float(stats.get("last_age_apply_ms"))),
                    fmt_ms(safe_float(stats.get("last_age_event_ms"))),
                ]
            )
        lines.append(
            md_table(
                ["venue", "samples", "max_cache_err", "max_ext_future", "last_age_apply_ms", "last_age_event_ms"],
                rows,
            )
        )
    else:
        lines.append("_No `component=runner_apply` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Runner Apply Truth Audit (WS_AUDIT)")
    if runner_apply_truth_stats:
        rows: list[list[str]] = []
        for venue, stats in sorted(runner_apply_truth_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_ext_apply_eligible_total"))),
                    fmt_int(safe_int(stats.get("max_ext_apply_same_tick_repeat_total"))),
                    fmt_int(safe_int(stats.get("max_ext_apply_frozen_total"))),
                    fmt_int(safe_int(stats.get("max_ext_apply_missing_metrics_total"))),
                    fmt_int(safe_int(stats.get("max_ext_freeze_nonpositive_spread"))),
                    fmt_int(safe_int(stats.get("max_ext_freeze_rel_spread"))),
                    fmt_int(safe_int(stats.get("max_ext_freeze_mid_jump"))),
                    fmt_ms(safe_float(stats.get("last_age_apply_ms"))),
                    fmt_ms(safe_float(stats.get("last_age_event_ms"))),
                    fmt_ms(safe_float(stats.get("last_venue_state_stale_ms"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "max_ext_apply_eligible_total",
                    "max_ext_apply_same_tick_repeat_total",
                    "max_ext_apply_frozen_total",
                    "max_ext_apply_missing_metrics_total",
                    "max_ext_freeze_nonpositive_spread",
                    "max_ext_freeze_rel_spread",
                    "max_ext_freeze_mid_jump",
                    "last_age_apply_ms",
                    "last_age_event_ms",
                    "last_venue_state_stale_ms",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=runner_apply_truth` entries found in run.log._")
    lines.append("")

    lines.append("## Extended Stale Defect Classification")
    total_episodes = safe_int(extended_defect_summary.get("total")) or 0
    if total_episodes > 0:
        lines.append(
            "- dominant defect: "
            f"`{extended_defect_summary.get('dominant_class', 'unclassified')}` "
            f"at `{safe_float(extended_defect_summary.get('confidence_pct')) or 0.0:.1f}%` confidence "
            f"across `{total_episodes}` stale episodes"
        )
        rows = []
        counts = extended_defect_summary.get("counts", {})
        if isinstance(counts, dict):
            for defect_class in EXTENDED_DEFECT_CLASSES:
                rows.append(
                    [
                        defect_class,
                        fmt_int(safe_int(counts.get(defect_class))),
                    ]
                )
        lines.append(md_table(["defect_class", "episode_count"], rows))
    else:
        lines.append("_No Extended stale episodes were classified in run.log._")
    lines.append("")

    lines.append("## REST Monitor Audit (WS_AUDIT)")
    if rest_monitor_stats:
        rows = []
        for venue, stats in sorted(rest_monitor_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_rest_success_count"))),
                    fmt_int(safe_int(stats.get("max_rest_fail_count"))),
                    fmt_int(safe_int(stats.get("max_rest_inject_count"))),
                    fmt_int(safe_int(stats.get("max_rest_suppressed_count"))),
                    fmt_ms(safe_float(stats.get("last_age_ms"))),
                    fmt_int(safe_int(stats.get("last_threshold_ms"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "max_rest_success_count",
                    "max_rest_fail_count",
                    "max_rest_inject_count",
                    "max_rest_suppressed_count",
                    "last_age_ms",
                    "last_threshold_ms",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `subsystem=rest_monitor` entries found in run.log._")
    lines.append("")

    lines.append("## Arb Gate Audit (WS_AUDIT)")
    if arb_gate_stats:
        rows = []
        for venue, stats in sorted(arb_gate_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_gated_ticks"))),
                    fmt_int(safe_int(stats.get("last_apply_age_ms"))),
                    fmt_int(safe_int(stats.get("threshold_ms"))),
                ]
            )
        lines.append(
            md_table(
                ["venue", "samples", "max_gated_ticks", "last_apply_age_ms", "threshold_ms"],
                rows,
            )
        )
    else:
        lines.append("_No `subsystem=arb_gate` entries found in run.log._")
    lines.append("")

    lines.append("## HL PubQ Audit (WS_AUDIT)")
    if hl_pubq_stats:
        rows = []
        for venue, stats in sorted(hl_pubq_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_try_send_full"))),
                    fmt_int(safe_int(stats.get("max_pending_overwrite"))),
                    fmt_int(safe_int(stats.get("max_pending_lock_fail"))),
                    fmt_int(safe_int(stats.get("max_ts_zero_count"))),
                    fmt_int(safe_int(stats.get("max_ws_rx_age_ms"))),
                    fmt_int(safe_int(stats.get("max_data_rx_age_ms"))),
                    fmt_int(safe_int(stats.get("max_pub_age_ms"))),
                    fmt_int(safe_int(stats.get("max_book_age_ms"))),
                    fmt_int(safe_int(stats.get("max_pub_minus_book_age_ms"))),
                    fmt_int(safe_int(stats.get("max_send_block_max_ms"))),
                    fmt_int(safe_int(stats.get("max_send_block_gt_5ms"))),
                    fmt_int(safe_int(stats.get("max_send_block_gt_50ms"))),
                    fmt_int(safe_int(stats.get("max_send_block_gt_250ms"))),
                    fmt_int(safe_int(stats.get("max_forward_send_count"))),
                    fmt_int(safe_int(stats.get("max_forward_send_err_count"))),
                    fmt_int(safe_int(stats.get("max_coalesced_drop_count"))),
                    fmt_int(safe_int(stats.get("max_pending_take_count"))),
                    fmt_int(safe_int(stats.get("max_ts_missing_or_zero_count"))),
                    fmt_int(safe_int(stats.get("max_ts_clamped_past_skew_count"))),
                    fmt_int(safe_int(stats.get("max_ts_clamped_future_skew_count"))),
                    fmt_int(safe_int(stats.get("max_ts_policy_enabled"))),
                    fmt_int(safe_int(stats.get("max_ts_policy_applied_count"))),
                    fmt_int(safe_int(stats.get("max_ts_kept_exchange_count"))),
                    fmt_int(safe_int(stats.get("max_ts_past_skew_max_ms"))),
                    fmt_int(safe_int(stats.get("max_ts_future_skew_max_ms"))),
                    fmt_int(safe_int(stats.get("max_queued_hiwater"))),
                    fmt_int(safe_int(stats.get("max_queued_len"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "max_try_send_full",
                    "max_pending_overwrite",
                    "max_pending_lock_fail",
                    "max_ts_zero_count",
                    "max_ws_rx_age_ms",
                    "max_data_rx_age_ms",
                    "max_pub_age_ms",
                    "max_book_age_ms",
                    "max_pub_minus_book_age_ms",
                    "max_send_block_max_ms",
                    "max_send_block_gt_5ms",
                    "max_send_block_gt_50ms",
                    "max_send_block_gt_250ms",
                    "max_forward_send_count",
                    "max_forward_send_err_count",
                    "max_coalesced_drop_count",
                    "max_pending_take_count",
                    "max_ts_missing_or_zero_count",
                    "max_ts_clamped_past_skew_count",
                    "max_ts_clamped_future_skew_count",
                    "max_ts_policy_enabled",
                    "max_ts_policy_applied_count",
                    "max_ts_kept_exchange_count",
                    "max_ts_past_skew_max_ms",
                    "max_ts_future_skew_max_ms",
                    "max_queued_hiwater",
                    "max_queued_len",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=hl_pubq` entries found in run.log._")
    lines.append("")

    lines.append("## Extended WS Msg Audit (WS_AUDIT)")
    if extended_ws_msg_stats:
        rows = []
        for venue, stats in sorted(extended_ws_msg_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    str(stats.get("last_reason", "n/a")),
                    str(stats.get("last_last_frame_kind", "n/a")),
                    str(stats.get("last_last_data_kind", "n/a")),
                    fmt_int(safe_int(stats.get("max_stale_ms"))),
                    fmt_int(safe_int(stats.get("max_parse_err"))),
                    fmt_int(safe_int(stats.get("max_publish_err"))),
                    fmt_int(safe_int(stats.get("max_max_gap_ms"))),
                    fmt_int(safe_int(stats.get("max_age_published_ms"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "last_reason",
                    "last_frame_kind",
                    "last_data_kind",
                    "max_stale_ms",
                    "max_parse_err",
                    "max_publish_err",
                    "max_max_gap_ms",
                    "max_age_published_ms",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=ws_msg` entries found in run.log._")
    lines.append("")

    lines.append("## Extended WS Read Timeout Audit (WS_AUDIT)")
    if extended_cfg_stats:
        rows = []
        for venue, stats in sorted(extended_cfg_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("last_extended_read_timeout_ms"))),
                    fmt_int(safe_int(stats.get("max_extended_read_timeout_ms"))),
                    fmt_int(safe_int(stats.get("last_extended_connect_book_timeout_ms"))),
                    fmt_int(safe_int(stats.get("max_extended_connect_book_timeout_ms"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "last_extended_read_timeout_ms",
                    "max_extended_read_timeout_ms",
                    "last_extended_connect_book_timeout_ms",
                    "max_extended_connect_book_timeout_ms",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `extended_read_timeout_ms` entries found in run.log._")
    lines.append("")

    lines.append("## Ping Audit (WS_AUDIT)")
    if ping_stats:
        rows = []
        for venue, stats in sorted(ping_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_ping_sent_count"))),
                    fmt_int(safe_int(stats.get("max_ping_send_fail_count"))),
                    str(stats.get("last_err", "n/a")),
                ]
            )
        lines.append(
            md_table(
                ["venue", "samples", "max_ping_sent_count", "max_ping_send_fail_count", "last_err"],
                rows,
            )
        )
    else:
        lines.append("_No ping audit entries found in run.log._")
    lines.append("")

    lines.append("## Aster Book Recovery Audit (WS_AUDIT)")
    if aster_book_recovery_stats:
        rows = []
        for stage_phase, stats in sorted(aster_book_recovery_stats.items()):
            rows.append(
                [
                    stage_phase,
                    fmt_int(safe_int(stats.get("samples"))),
                    str(stats.get("last_failure_class") or "n/a"),
                    fmt_int(safe_int(stats.get("max_http_status"))),
                    fmt_int(safe_int(stats.get("max_fail_streak"))),
                    fmt_int(safe_int(stats.get("max_cooldown_ms"))),
                    fmt_int(safe_int(stats.get("max_weight_1m"))),
                    fmt_int(safe_int(stats.get("max_buffered_before"))),
                    fmt_int(safe_int(stats.get("max_applied_count"))),
                    fmt_int(safe_int(stats.get("max_snap_id"))),
                    fmt_int(safe_int(stats.get("max_update_end"))),
                    fmt_int(safe_int(stats.get("max_current_last"))),
                    fmt_int(safe_int(stats.get("max_stale_ms"))),
                    fmt_int(safe_int(stats.get("max_anchor_age_ms"))),
                    fmt_int(safe_int(stats.get("max_book_age_ms"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "stage/phase",
                    "samples",
                    "last_failure_class",
                    "max_http_status",
                    "max_fail_streak",
                    "max_cooldown_ms",
                    "max_weight_1m",
                    "max_buffered_before",
                    "max_applied_count",
                    "max_snap_id",
                    "max_update_end",
                    "max_current_last",
                    "max_stale_ms",
                    "max_anchor_age_ms",
                    "max_book_age_ms",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `component=book_recovery` entries found in run.log._")
    lines.append("")

    lines.append("## Lighter Timestamp Fallback Audit (WS_AUDIT)")
    if lighter_ts_fallback_stats:
        rows = []
        for venue, stats in sorted(lighter_ts_fallback_stats.items()):
            rows.append(
                [
                    venue,
                    fmt_int(safe_int(stats.get("samples"))),
                    fmt_int(safe_int(stats.get("max_lighter_ts_fallback_count"))),
                    str(stats.get("last_context", "n/a")),
                    str(stats.get("last_raw_ts", "n/a")),
                    fmt_int(safe_int(stats.get("max_fallback_ts_ms"))),
                ]
            )
        lines.append(
            md_table(
                [
                    "venue",
                    "samples",
                    "max_lighter_ts_fallback_count",
                    "last_context",
                    "last_raw_ts",
                    "max_fallback_ts_ms",
                ],
                rows,
            )
        )
    else:
        lines.append("_No `lighter_ts_fallback_count` entries found in run.log._")
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


def build_metrics_payload(
    telemetry_summary: TelemetrySummary,
    extended_cfg_stats: dict[str, dict[str, Any]],
    extended_reconnect_policy_stats: dict[str, dict[str, Any]],
    extended_transport_gap_stats: dict[str, dict[str, Any]],
    extended_rest_seed_summary: dict[str, Any],
    extended_seed_bridge_summary: dict[str, Any],
    extended_control_frame_grace_summary: dict[str, Any],
    extended_session_hedge_summary: dict[str, Any],
    extended_backend_attach_fallback_summary: dict[str, Any],
    extended_post_publish_fallback_summary: dict[str, Any],
    extended_post_publish_gap_stage_summary: dict[str, Any],
    extended_stream_preference_summary: dict[str, Any],
    extended_bootstrap_timeout_stats: dict[str, dict[str, Any]],
    extended_bootstrap_churn_stats: dict[str, dict[str, Any]],
    extended_bootstrap_summary: dict[str, Any],
    extended_control_frame_before_data_summary: dict[str, Any],
    extended_socket_establishment_summary: dict[str, Any],
    extended_socket_role_progress_summary: dict[str, Any],
    extended_stream_kind_progress_summary: dict[str, Any],
    extended_defect_summary: dict[str, Any],
) -> dict[str, Any]:
    extended_bootstrap_stage_summary = summarize_extended_bootstrap_stages(
        extended_bootstrap_timeout_stats
    )
    extended_first_frame_timeout_summary = summarize_extended_first_frame_timeout(
        extended_bootstrap_timeout_stats
    )
    extended_first_data_timeout_summary = summarize_extended_first_data_timeout(
        extended_bootstrap_timeout_stats
    )
    extended_watchdog_bootstrap_transition_summary = (
        summarize_extended_watchdog_bootstrap_transition(extended_transport_gap_stats)
    )
    return {
        "schema_version": 1,
        "telemetry_summary": {
            "rows": telemetry_summary.rows,
            "first_tick": telemetry_summary.first_tick,
            "last_tick": telemetry_summary.last_tick,
            "first_ts_ms": telemetry_summary.first_ts_ms,
            "last_ts_ms": telemetry_summary.last_ts_ms,
        },
        "extended_cfg_stats": extended_cfg_stats,
        "extended_reconnect_policy_stats": extended_reconnect_policy_stats,
        "extended_transport_gap_stats": extended_transport_gap_stats,
        "extended_rest_seed_summary": extended_rest_seed_summary,
        "extended_rest_seed_failures": {
            "by_status": extended_rest_seed_summary.get("failures_by_status", {}),
            "by_http_status": extended_rest_seed_summary.get("failures_by_http_status", {}),
        },
        "extended_seed_bridge_summary": extended_seed_bridge_summary,
        "extended_control_frame_grace_summary": extended_control_frame_grace_summary,
        "extended_session_hedge_summary": extended_session_hedge_summary,
        "extended_backend_attach_fallback_summary": extended_backend_attach_fallback_summary,
        "extended_post_publish_fallback_summary": extended_post_publish_fallback_summary,
        "extended_post_publish_gap_stage_summary": extended_post_publish_gap_stage_summary,
        "extended_stream_preference_summary": extended_stream_preference_summary,
        "extended_bootstrap_timeout_stats": extended_bootstrap_timeout_stats,
        "extended_bootstrap_churn_stats": extended_bootstrap_churn_stats,
        "extended_bootstrap_summary": extended_bootstrap_summary,
        "extended_bootstrap_reason_summary": extended_bootstrap_summary,
        "extended_bootstrap_stage_summary": extended_bootstrap_stage_summary,
        "extended_first_frame_timeout_summary": extended_first_frame_timeout_summary,
        "extended_first_data_timeout_summary": extended_first_data_timeout_summary,
        "extended_control_frame_before_data_summary": extended_control_frame_before_data_summary,
        "extended_pre_first_data_shape_summary": extended_control_frame_before_data_summary,
        "extended_socket_establishment_summary": extended_socket_establishment_summary,
        "extended_socket_role_progress_summary": extended_socket_role_progress_summary,
        "extended_stream_kind_progress_summary": extended_stream_kind_progress_summary,
        "extended_watchdog_bootstrap_transition_summary": extended_watchdog_bootstrap_transition_summary,
        "extended_defect_summary": extended_defect_summary,
    }


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
        (
            audit_reconnect,
            signature_reconnect,
            market_publisher,
            runner_apply_stats,
            runner_apply_truth_stats,
            rest_monitor_stats,
            arb_gate_stats,
            hl_pubq_stats,
            extended_ws_msg_stats,
            extended_cfg_stats,
            extended_reconnect_policy_stats,
            extended_transport_gap_stats,
            extended_rest_seed_summary,
            extended_seed_bridge_summary,
            extended_control_frame_grace_summary,
            extended_session_hedge_stats,
            extended_backend_attach_fallback_summary,
            extended_bootstrap_timeout_stats,
            extended_bootstrap_churn_stats,
            extended_bootstrap_summary,
            ping_stats,
            lighter_ts_fallback_stats,
            aster_book_recovery_stats,
            extended_defect_summary,
            extended_control_frame_before_data_summary,
            extended_socket_establishment_summary,
            extended_socket_role_progress_summary,
            extended_stream_kind_progress_summary,
            extended_post_publish_fallback_summary,
            extended_post_publish_gap_stage_summary,
            extended_stream_preference_summary,
        ) = parse_run_log(run_log_path)
        cap_hits = parse_market_rx_stats(market_rx_path) if market_rx_path.exists() else None
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    extended_session_hedge_summary = summarize_extended_session_hedge(
        extended_session_hedge_stats
    )
    report = build_report(
        out_dir=out_dir,
        telemetry_summary=telemetry_summary,
        apply_values=apply_values,
        event_values=event_values,
        max_plateaus=max_plateaus,
        audit_reconnect=audit_reconnect,
        signature_reconnect=signature_reconnect,
        market_publisher_counters=market_publisher,
        runner_apply_stats=runner_apply_stats,
        runner_apply_truth_stats=runner_apply_truth_stats,
        rest_monitor_stats=rest_monitor_stats,
        arb_gate_stats=arb_gate_stats,
        hl_pubq_stats=hl_pubq_stats,
        extended_ws_msg_stats=extended_ws_msg_stats,
        extended_cfg_stats=extended_cfg_stats,
        extended_reconnect_policy_stats=extended_reconnect_policy_stats,
        extended_transport_gap_stats=extended_transport_gap_stats,
        extended_rest_seed_summary=extended_rest_seed_summary,
        extended_seed_bridge_summary=extended_seed_bridge_summary,
        extended_control_frame_grace_summary=extended_control_frame_grace_summary,
        extended_session_hedge_summary=extended_session_hedge_summary,
        extended_backend_attach_fallback_summary=extended_backend_attach_fallback_summary,
        extended_post_publish_fallback_summary=extended_post_publish_fallback_summary,
        extended_post_publish_gap_stage_summary=extended_post_publish_gap_stage_summary,
        extended_stream_preference_summary=extended_stream_preference_summary,
        extended_bootstrap_timeout_stats=extended_bootstrap_timeout_stats,
        extended_bootstrap_churn_stats=extended_bootstrap_churn_stats,
        extended_bootstrap_summary=extended_bootstrap_summary,
        extended_control_frame_before_data_summary=extended_control_frame_before_data_summary,
        extended_socket_establishment_summary=extended_socket_establishment_summary,
        extended_socket_role_progress_summary=extended_socket_role_progress_summary,
        extended_stream_kind_progress_summary=extended_stream_kind_progress_summary,
        ping_stats=ping_stats,
        lighter_ts_fallback_stats=lighter_ts_fallback_stats,
        aster_book_recovery_stats=aster_book_recovery_stats,
        extended_defect_summary=extended_defect_summary,
        cap_hits_summary=cap_hits,
    )

    report_path = out_dir / "ws_soak_report.md"
    report_path.write_text(report, encoding="utf-8")
    metrics_path = out_dir / "ws_soak_metrics.json"
    metrics_path.write_text(
        json.dumps(
            build_metrics_payload(
                telemetry_summary=telemetry_summary,
                extended_cfg_stats=extended_cfg_stats,
                extended_reconnect_policy_stats=extended_reconnect_policy_stats,
                extended_transport_gap_stats=extended_transport_gap_stats,
                extended_rest_seed_summary=extended_rest_seed_summary,
                extended_seed_bridge_summary=extended_seed_bridge_summary,
                extended_control_frame_grace_summary=extended_control_frame_grace_summary,
                extended_session_hedge_summary=extended_session_hedge_summary,
                extended_backend_attach_fallback_summary=extended_backend_attach_fallback_summary,
                extended_post_publish_fallback_summary=extended_post_publish_fallback_summary,
                extended_post_publish_gap_stage_summary=extended_post_publish_gap_stage_summary,
                extended_stream_preference_summary=extended_stream_preference_summary,
                extended_bootstrap_timeout_stats=extended_bootstrap_timeout_stats,
                extended_bootstrap_churn_stats=extended_bootstrap_churn_stats,
                extended_bootstrap_summary=extended_bootstrap_summary,
                extended_control_frame_before_data_summary=extended_control_frame_before_data_summary,
                extended_socket_establishment_summary=extended_socket_establishment_summary,
                extended_socket_role_progress_summary=extended_socket_role_progress_summary,
                extended_stream_kind_progress_summary=extended_stream_kind_progress_summary,
                extended_defect_summary=extended_defect_summary,
            ),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report, end="")
    print(f"\n_report saved to `{report_path}`_")
    print(f"_metrics saved to `{metrics_path}`_")

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
