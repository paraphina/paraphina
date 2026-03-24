#!/usr/bin/env python3
"""Low-impact unattended guard for a running Paraphina live canary.

The guard polls health and systemd state on a timer for a bounded window.
If the live service degrades, it restores shadow mode and runs venue cleanup
only when the post-rollback venue audit is dirty.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(level: str, message: str) -> None:
    print(f"{iso_utc()} {level} {message}", flush=True)


def run_cmd(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def read_health(url: str, timeout_sec: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return None


def read_systemd(service: str) -> dict[str, str]:
    result = run_cmd(
        [
            "systemctl",
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "NRestarts",
        ]
    )
    payload: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key] = value
    return payload


def run_venue_audit(audit_script: Path, env_file: Path, max_open_orders: int) -> dict[str, Any] | None:
    result = run_cmd(
        [
            sys.executable,
            str(audit_script),
            "--env-file",
            str(env_file),
            "--position-tol-base",
            "0.0025",
            "--max-open-orders",
            str(max_open_orders),
        ],
        timeout=120.0,
    )
    if result.returncode != 0:
        log("WARN", f"venue_audit_failed rc={result.returncode} stderr={result.stderr.strip()!r}")
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log("WARN", f"venue_audit_invalid_json stdout={result.stdout[:400]!r}")
        return None
    return data if isinstance(data, dict) else None


def audit_is_clean(audit: dict[str, Any] | None) -> bool:
    if not audit:
        return False
    if audit.get("violations"):
        return False
    for item in audit.get("results", []):
        if not item.get("ok", False):
            return False
        try:
            position = abs(float(item.get("position_base", 0.0)))
        except (TypeError, ValueError):
            return False
        if position > 0.0025:
            return False
        try:
            open_orders = int(item.get("open_order_count", 0))
        except (TypeError, ValueError):
            return False
        if open_orders > 0:
            return False
    return True


def build_cleanup_args(cleanup_bin: Path, env_file: Path, audit: dict[str, Any] | None) -> list[str]:
    args = [str(cleanup_bin), "--env-file", str(env_file), "--settle-ms", "2000"]
    if not audit:
        return args
    per_venue = {str(item.get("venue")): item for item in audit.get("results", [])}
    for venue, flag in (
        ("lighter", "--lighter-pos"),
        ("extended", "--extended-pos"),
        ("aster", "--aster-pos"),
        ("paradex", "--paradex-pos"),
    ):
        item = per_venue.get(venue)
        if not item:
            continue
        try:
            pos = float(item.get("position_base", 0.0))
        except (TypeError, ValueError):
            pos = 0.0
        args.extend([flag, str(pos)])
    return args


def wait_for_shadow(url: str, timeout_sec: float, deadline_sec: float) -> bool:
    deadline = time.time() + deadline_sec
    while time.time() < deadline:
        health = read_health(url, timeout_sec)
        if health and health.get("healthy") and health.get("ready") and health.get("trade_mode") == "shadow":
            return True
        time.sleep(2.0)
    return False


def intervene(
    reason: str,
    config_manager: Path,
    config_dir: Path,
    env_file: Path,
    audit_script: Path,
    cleanup_bin: Path,
    health_url: str,
    health_timeout_sec: float,
) -> int:
    log("CRITICAL", f"triggered_intervention reason={reason}")
    result = run_cmd(
        [
            "sudo",
            sys.executable,
            str(config_manager),
            "--config-dir",
            str(config_dir),
            "activate",
            str(env_file),
            "--stage",
            "shadow",
            "--restart",
        ],
        timeout=180.0,
    )
    log("INFO", f"shadow_activation rc={result.returncode}")
    if result.stdout.strip():
        log("INFO", f"shadow_activation_stdout={result.stdout.strip()!r}")
    if result.stderr.strip():
        log("WARN", f"shadow_activation_stderr={result.stderr.strip()!r}")

    if not wait_for_shadow(health_url, health_timeout_sec, deadline_sec=90.0):
        log("CRITICAL", "shadow_health_recovery_failed")
        return 2

    audit = run_venue_audit(audit_script, env_file, max_open_orders=0)
    if audit_is_clean(audit):
        log("INFO", "post_rollback_venue_audit_clean")
        return 0

    cleanup_args = build_cleanup_args(cleanup_bin, env_file, audit)
    cleanup = run_cmd(cleanup_args, timeout=180.0)
    log("INFO", f"live_cleanup rc={cleanup.returncode}")
    if cleanup.stdout.strip():
        log("INFO", f"live_cleanup_stdout={cleanup.stdout.strip()!r}")
    if cleanup.stderr.strip():
        log("WARN", f"live_cleanup_stderr={cleanup.stderr.strip()!r}")

    confirm = run_venue_audit(audit_script, env_file, max_open_orders=0)
    if audit_is_clean(confirm):
        log("INFO", "post_cleanup_venue_audit_clean")
        return 0
    log("CRITICAL", f"post_cleanup_venue_audit_dirty payload={json.dumps(confirm or {}, separators=(',', ':'))}")
    return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous guard for a running Paraphina live canary")
    parser.add_argument("--duration-sec", type=int, default=10800)
    parser.add_argument("--poll-sec", type=float, default=15.0)
    parser.add_argument("--summary-sec", type=float, default=300.0)
    parser.add_argument("--health-url", default="http://127.0.0.1:9898/health/detail")
    parser.add_argument("--health-timeout-sec", type=float, default=2.0)
    parser.add_argument("--service", default="paraphina_live")
    parser.add_argument("--config-dir", default="/etc/paraphina")
    parser.add_argument("--env-file", default="/etc/paraphina/current.env")
    parser.add_argument("--config-manager", default="/home/ubuntu/paraphina/deploy/config_manager.py")
    parser.add_argument("--audit-script", default="/home/ubuntu/paraphina/tools/live_venue_audit.py")
    parser.add_argument("--cleanup-bin", default="/home/ubuntu/paraphina/target/release/live_cleanup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    deadline = start + float(args.duration_sec)
    last_summary = 0.0
    unreachable_streak = 0
    unhealthy_streak = 0

    config_dir = Path(args.config_dir)
    env_file = Path(args.env_file)
    config_manager = Path(args.config_manager)
    audit_script = Path(args.audit_script)
    cleanup_bin = Path(args.cleanup_bin)

    log("INFO", f"guard_started duration_sec={args.duration_sec} poll_sec={args.poll_sec}")
    while time.time() < deadline:
        health = read_health(args.health_url, args.health_timeout_sec)
        systemd = read_systemd(args.service)
        active = systemd.get("ActiveState", "unknown")
        substate = systemd.get("SubState", "unknown")
        restarts = systemd.get("NRestarts", "unknown")

        if health is None:
            unreachable_streak += 1
            log("WARN", f"health_unreachable streak={unreachable_streak}")
        else:
            unreachable_streak = 0
            healthy = bool(health.get("healthy"))
            ready = bool(health.get("ready"))
            if healthy and ready and health.get("trade_mode") == "live":
                unhealthy_streak = 0
            else:
                unhealthy_streak += 1

        now = time.time()
        if now - last_summary >= args.summary_sec:
            if health is None:
                log("INFO", f"summary active={active}/{substate} restarts={restarts} health=unreachable")
            else:
                log(
                    "INFO",
                    "summary "
                    f"active={active}/{substate} restarts={restarts} "
                    f"trade_mode={health.get('trade_mode')} healthy={health.get('healthy')} ready={health.get('ready')} "
                    f"kills={health.get('kill_events_present')} reconcile={health.get('reconcile_mismatch_count')} "
                    f"ticks={health.get('tick_count')} uptime={health.get('uptime_seconds')}",
                )
            last_summary = now

        reason: str | None = None
        if active != "active" or substate != "running":
            reason = f"systemd_state={active}/{substate}"
        elif health is None and unreachable_streak >= 3:
            reason = "health_unreachable_3x"
        elif health is not None:
            if health.get("kill_events_present"):
                reason = "kill_events_present"
            elif int(health.get("reconcile_mismatch_count", 0)) > 0:
                reason = f"reconcile_mismatch_count={health.get('reconcile_mismatch_count')}"
            elif int(restarts) > 0:
                reason = f"service_restarts={restarts}"
            elif unhealthy_streak >= 3:
                reason = (
                    f"unhealthy_streak=3 trade_mode={health.get('trade_mode')} "
                    f"healthy={health.get('healthy')} ready={health.get('ready')}"
                )

        if reason:
            return intervene(
                reason=reason,
                config_manager=config_manager,
                config_dir=config_dir,
                env_file=env_file,
                audit_script=audit_script,
                cleanup_bin=cleanup_bin,
                health_url=args.health_url,
                health_timeout_sec=args.health_timeout_sec,
            )

        time.sleep(args.poll_sec)

    log("INFO", "guard_completed_without_intervention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
