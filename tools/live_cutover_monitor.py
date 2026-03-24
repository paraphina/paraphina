#!/usr/bin/env python3
"""Monitor Paraphina launch-stage telemetry through a scheduled cutover.

This is designed for high-stakes first-capital launches:
- immediate alerts on critical telemetry/health changes
- regular summary lines at a low cadence (default 5 minutes)
- explicit tracking of the scheduled paper -> live cutover time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_cutover_utc(raw: str) -> datetime:
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def log_line(level: str, message: str) -> None:
    print(f"{iso_utc()} {level} {message}", flush=True)


def read_health_detail(url: str, timeout_sec: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return None


def read_new_records(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return offset, []
    size = path.stat().st_size
    if size < offset:
        offset = 0
    records: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
        offset = fh.tell()
    if not data:
        return offset, records
    for raw in data.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return offset, records


def read_last_record(path: Path, tail_bytes: int = 131072) -> tuple[int, dict[str, Any] | None]:
    if not path.exists():
        return 0, None
    size = path.stat().st_size
    if size == 0:
        return 0, None
    with path.open("rb") as fh:
        if size > tail_bytes:
            fh.seek(size - tail_bytes)
        data = fh.read()
    for raw in reversed(data.splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return size, obj
    return size, None


@dataclass
class MonitorState:
    offset: int = 0
    last_record: dict[str, Any] | None = None
    last_exec_mode: str | None = None
    last_kill_switch: bool | None = None
    last_kill_reason: str | None = None
    last_healthy_count: int | None = None
    last_risk_regime: str | None = None
    last_reconcile_drift_count: int | None = None
    last_ready: bool | None = None
    last_health_error_count: int | None = None
    last_health_reconcile_count: int | None = None
    last_summary_at: float = 0.0
    missed_cutover_alerted: bool = False
    health_unreachable_alerted: bool = False
    telemetry_stale_alerted: bool = False


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def seed_state_from_record(state: MonitorState, record: dict[str, Any]) -> None:
    state.last_record = record
    state.last_exec_mode = str(record.get("execution_mode", "n/a"))
    state.last_kill_switch = bool(record.get("kill_switch", False))
    state.last_kill_reason = str(record.get("kill_reason", "n/a"))
    state.last_healthy_count = safe_int(record.get("healthy_venues_used_count"))
    state.last_risk_regime = str(record.get("risk_regime", "n/a"))
    state.last_reconcile_drift_count = len(record.get("reconcile_drift") or [])


def summarize(record: dict[str, Any], health: dict[str, Any] | None) -> str:
    healthy_count = safe_int(record.get("healthy_venues_used_count"))
    parts = [
        f"tick={record.get('t', 'n/a')}",
        f"mode={record.get('execution_mode', 'n/a')}",
        f"risk={record.get('risk_regime', 'n/a')}",
        f"kill={record.get('kill_switch', False)}",
        f"kill_reason={record.get('kill_reason', 'n/a')}",
        f"healthy={healthy_count if healthy_count is not None else 'n/a'}",
        f"would_send={record.get('would_send_orders_count', 'n/a')}",
    ]
    if health:
        parts.extend(
            [
                f"ready={health.get('ready', 'n/a')}",
                f"tick_age_ms={health.get('tick_age_ms', 'n/a')}",
                f"errors={health.get('error_count', 'n/a')}",
                f"reconcile_mismatch={health.get('reconcile_mismatch_count', 'n/a')}",
                f"kill_events={health.get('kill_events_present', 'n/a')}",
            ]
        )
    return " ".join(parts)


def emit_record_alerts(
    state: MonitorState,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    if not records:
        return None, False

    latest = records[-1]
    saw_critical = False
    state.last_record = latest

    exec_mode = str(latest.get("execution_mode", "n/a"))
    kill_switch = bool(latest.get("kill_switch", False))
    kill_reason = str(latest.get("kill_reason", "n/a"))
    healthy_count = safe_int(latest.get("healthy_venues_used_count"))
    risk_regime = str(latest.get("risk_regime", "n/a"))
    reconcile_drift_count = len(latest.get("reconcile_drift") or [])

    if exec_mode != state.last_exec_mode:
        previous = state.last_exec_mode or "unknown"
        log_line("ALERT", f"execution_mode_changed from={previous} to={exec_mode}")
        state.last_exec_mode = exec_mode

    if kill_switch and not state.last_kill_switch:
        log_line("CRITICAL", f"kill_switch_triggered reason={kill_reason}")
        saw_critical = True
    state.last_kill_switch = kill_switch
    state.last_kill_reason = kill_reason

    if healthy_count is not None and healthy_count != state.last_healthy_count:
        log_line("INFO", f"healthy_venues_used_count changed to={healthy_count}")
        if healthy_count < 2:
            log_line("CRITICAL", f"healthy_venues_used_count critical value={healthy_count}")
            saw_critical = True
        elif healthy_count < 4:
            log_line("ALERT", f"healthy_venues_used_count degraded value={healthy_count}")
        state.last_healthy_count = healthy_count

    if risk_regime != state.last_risk_regime:
        log_line("INFO", f"risk_regime changed to={risk_regime}")
        if risk_regime not in {"Normal", "n/a"}:
            log_line("ALERT", f"risk_regime non_normal value={risk_regime}")
        state.last_risk_regime = risk_regime

    if reconcile_drift_count != state.last_reconcile_drift_count:
        if reconcile_drift_count > 0:
            log_line("ALERT", f"reconcile_drift detected count={reconcile_drift_count}")
        elif state.last_reconcile_drift_count and state.last_reconcile_drift_count > 0:
            log_line("INFO", "reconcile_drift cleared")
        state.last_reconcile_drift_count = reconcile_drift_count
    return latest, saw_critical


def emit_health_alerts(state: MonitorState, health: dict[str, Any] | None) -> None:
    if health is None:
        if not state.health_unreachable_alerted:
            log_line("WARN", "health_detail_unreachable")
            state.health_unreachable_alerted = True
        return

    state.health_unreachable_alerted = False
    ready = bool(health.get("ready", False))
    if state.last_ready is None or ready != state.last_ready:
        log_line("INFO", f"health_ready changed to={ready}")
        if not ready:
            log_line("ALERT", "health_ready=false")
        state.last_ready = ready

    error_count = safe_int(health.get("error_count"))
    if error_count is not None and error_count != state.last_health_error_count:
        log_line("INFO", f"health_error_count changed to={error_count}")
        if error_count > 0:
            log_line("ALERT", f"health_error_count nonzero value={error_count}")
        state.last_health_error_count = error_count

    reconcile_count = safe_int(health.get("reconcile_mismatch_count"))
    if reconcile_count is not None and reconcile_count != state.last_health_reconcile_count:
        log_line("INFO", f"health_reconcile_mismatch_count changed to={reconcile_count}")
        if reconcile_count > 0:
            log_line("CRITICAL", f"health_reconcile_mismatch_count nonzero value={reconcile_count}")
        state.last_health_reconcile_count = reconcile_count

    if bool(health.get("kill_events_present", False)):
        log_line("CRITICAL", "health_kill_events_present=true")

    tick_age_ms = safe_int(health.get("tick_age_ms"))
    if tick_age_ms is not None and tick_age_ms > 30_000:
        log_line("ALERT", f"tick_age_ms high value={tick_age_ms}")


def emit_cutover_alerts(
    state: MonitorState,
    latest: dict[str, Any] | None,
    cutover_utc: datetime,
    grace_sec: int,
) -> None:
    now = utc_now()
    seconds_to_cutover = int((cutover_utc - now).total_seconds())
    if seconds_to_cutover > 0:
        if seconds_to_cutover in {1800, 1200, 900, 600, 300, 120, 60, 30}:
            log_line(
                "INFO",
                f"cutover_countdown seconds_to_cutover={seconds_to_cutover}",
            )
        return
    exec_mode = None
    if latest:
        exec_mode = str(latest.get("execution_mode", "n/a"))
    if exec_mode == "paper" and -seconds_to_cutover >= grace_sec and not state.missed_cutover_alerted:
        log_line(
            "CRITICAL",
            "scheduled_cutover_missed execution_mode=paper "
            f"cutover_utc={iso_utc(cutover_utc)} grace_sec={grace_sec}",
        )
        state.missed_cutover_alerted = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Paraphina cutover telemetry")
    parser.add_argument(
        "--telemetry",
        default="/var/lib/paraphina/out/telemetry.jsonl",
        help="Telemetry JSONL path",
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:9898/health/detail",
        help="health/detail URL",
    )
    parser.add_argument(
        "--poll-sec",
        type=int,
        default=30,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--summary-sec",
        type=int,
        default=300,
        help="Summary interval in seconds",
    )
    parser.add_argument(
        "--health-timeout-sec",
        type=float,
        default=2.0,
        help="health/detail timeout",
    )
    parser.add_argument(
        "--cutover-utc",
        required=True,
        help="Scheduled cutover UTC timestamp, e.g. 2026-03-14T01:52:00Z",
    )
    parser.add_argument(
        "--cutover-grace-sec",
        type=int,
        default=120,
        help="Grace window after cutover before alerting if still in paper mode",
    )
    args = parser.parse_args()

    telemetry_path = Path(args.telemetry)
    cutover_utc = parse_cutover_utc(args.cutover_utc)
    state = MonitorState()
    state.offset, bootstrap_record = read_last_record(telemetry_path)
    if bootstrap_record:
        seed_state_from_record(state, bootstrap_record)

    log_line(
        "INFO",
        f"monitor_started telemetry={telemetry_path} cutover_utc={iso_utc(cutover_utc)} "
        f"poll_sec={args.poll_sec} summary_sec={args.summary_sec}",
    )
    if bootstrap_record:
        log_line(
            "INFO",
            "monitor_bootstrap "
            f"tick={bootstrap_record.get('t', 'n/a')} "
            f"mode={state.last_exec_mode} "
            f"risk={state.last_risk_regime} "
            f"kill={state.last_kill_switch} "
            f"healthy={state.last_healthy_count if state.last_healthy_count is not None else 'n/a'}",
        )

    while True:
        now_monotonic = time.monotonic()
        state.offset, records = read_new_records(telemetry_path, state.offset)
        latest, _ = emit_record_alerts(state, records)
        health = read_health_detail(args.health_url, args.health_timeout_sec)
        emit_health_alerts(state, health)

        if telemetry_path.exists():
            age_sec = time.time() - telemetry_path.stat().st_mtime
            if age_sec > max(60, args.poll_sec * 2):
                if not state.telemetry_stale_alerted:
                    log_line("CRITICAL", f"telemetry_stale age_sec={age_sec:.1f}")
                    state.telemetry_stale_alerted = True
            else:
                state.telemetry_stale_alerted = False

        emit_cutover_alerts(state, latest or state.last_record, cutover_utc, args.cutover_grace_sec)

        if state.last_record and (now_monotonic - state.last_summary_at >= args.summary_sec):
            log_line("SUMMARY", summarize(state.last_record, health))
            state.last_summary_at = now_monotonic

        time.sleep(max(1, args.poll_sec))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log_line("INFO", "monitor_stopped")
        raise SystemExit(130)
