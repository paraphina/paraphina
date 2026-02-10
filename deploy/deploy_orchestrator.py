#!/usr/bin/env python3
"""
deploy_orchestrator.py — Graduated auto-deploy pipeline for Paraphina.

Progresses a promoted config through staged soak periods with health gates:
  Stage 1: Shadow soak  — observe-only, verify telemetry flowing
  Stage 2: Paper soak   — simulated fills, verify PnL/drawdown within budget
  Stage 3: Canary live   — real trading with tight limits
  Stage 4: Full live     — production with normal limits

Any gate failure triggers immediate rollback to the previous known-good config.

Usage:
    # Full graduated deploy (default soak durations)
    python3 deploy/deploy_orchestrator.py deploy <promoted.env> \\
        --config-dir /etc/paraphina \\
        --health-url http://127.0.0.1:9898

    # Deploy with custom soak durations (in seconds)
    python3 deploy/deploy_orchestrator.py deploy <promoted.env> \\
        --shadow-soak 300 --paper-soak 600 --canary-soak 900

    # Dry-run: print what would happen without restarting
    python3 deploy/deploy_orchestrator.py deploy <promoted.env> --dry-run

    # Deploy only through paper (human approves canary->live)
    python3 deploy/deploy_orchestrator.py deploy <promoted.env> --stop-before canary

    # Rollback immediately
    python3 deploy/deploy_orchestrator.py rollback --config-dir /etc/paraphina

See ADR-001 (docs/adr/001-auto-deploy-policy.md) for design rationale.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ===========================================================================
# Constants
# ===========================================================================

DEFAULT_CONFIG_DIR = Path("/etc/paraphina")
DEFAULT_HEALTH_URL = "http://127.0.0.1:9898"
SERVICE_NAME = "paraphina_live"

# Default soak durations in seconds
DEFAULT_SHADOW_SOAK = 300   # 5 minutes
DEFAULT_PAPER_SOAK = 600    # 10 minutes
DEFAULT_CANARY_SOAK = 900   # 15 minutes

# Health polling interval in seconds
POLL_INTERVAL = 10

# Maximum time to wait for service to become healthy after restart
STARTUP_TIMEOUT = 60

# Tiers that are NOT eligible for auto-deploy
MANUAL_ONLY_TIERS = {"aggressive"}


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class StageConfig:
    """Configuration for a single deploy stage."""
    name: str
    trade_mode: str
    canary_mode: bool
    soak_seconds: int
    env_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeployResult:
    """Result of a deploy attempt."""
    success: bool
    final_stage: str
    stages_passed: List[str]
    failure_reason: Optional[str] = None
    rolled_back: bool = False
    duration_seconds: float = 0.0


@dataclass
class HealthSnapshot:
    """Parsed health detail from the /health/detail endpoint."""
    healthy: bool = False
    ready: bool = False
    uptime_seconds: int = 0
    tick_count: int = 0
    last_tick_ms: int = 0
    tick_age_ms: int = -1
    error_count: int = 0
    reconcile_mismatch_count: int = 0
    kill_events_present: bool = False
    trade_mode: str = ""
    config_id: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HealthSnapshot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ===========================================================================
# Health checking
# ===========================================================================

def fetch_health_detail(base_url: str, timeout: float = 5.0) -> Optional[HealthSnapshot]:
    """Fetch /health/detail JSON from the running service."""
    import urllib.request
    import urllib.error

    url = f"{base_url}/health/detail"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return HealthSnapshot.from_dict(data)
            return None
    except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
        return None


def wait_for_healthy(
    base_url: str,
    timeout: int = STARTUP_TIMEOUT,
    poll_interval: int = 3,
    log_fn=print,
) -> bool:
    """Wait for the service to become healthy after restart."""
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        snap = fetch_health_detail(base_url)
        if snap and snap.healthy and snap.tick_count > 0:
            log_fn(f"  Service healthy after {attempt} polls (uptime={snap.uptime_seconds}s, ticks={snap.tick_count})")
            return True
        time.sleep(poll_interval)
    log_fn(f"  Service did not become healthy within {timeout}s")
    return False


# ===========================================================================
# Stage health predicates
# ===========================================================================

def check_shadow_health(snap: HealthSnapshot, soak_elapsed: float) -> Optional[str]:
    """Shadow stage: process alive, telemetry flowing, no kill events.

    Returns None if healthy, or a failure reason string.
    """
    if not snap.healthy:
        return "service reports unhealthy"
    if snap.kill_events_present:
        return "kill events detected during shadow soak"
    # After 30 seconds of soak, we expect ticks to be flowing
    if soak_elapsed > 30 and snap.tick_count < 1:
        return f"no ticks after {soak_elapsed:.0f}s of shadow soak"
    # Tick staleness: if last tick is more than 30s old, something is wrong
    if snap.tick_age_ms > 30_000:
        return f"tick stale: last tick {snap.tick_age_ms}ms ago"
    return None


def check_paper_health(snap: HealthSnapshot, soak_elapsed: float) -> Optional[str]:
    """Paper stage: no kills, ticks flowing, errors reasonable.

    Returns None if healthy, or a failure reason string.
    """
    if not snap.healthy:
        return "service reports unhealthy"
    if snap.kill_events_present:
        return "kill events detected during paper soak"
    if snap.tick_age_ms > 30_000:
        return f"tick stale: last tick {snap.tick_age_ms}ms ago"
    # More than 10 errors is suspicious during paper trading
    if snap.error_count > 10:
        return f"excessive errors during paper soak: {snap.error_count}"
    return None


def check_canary_health(snap: HealthSnapshot, soak_elapsed: float) -> Optional[str]:
    """Canary stage: no kills, no reconciliation drift, ticks flowing.

    Returns None if healthy, or a failure reason string.
    """
    if not snap.healthy:
        return "service reports unhealthy"
    if snap.kill_events_present:
        return "kill events detected during canary soak"
    if snap.tick_age_ms > 30_000:
        return f"tick stale: last tick {snap.tick_age_ms}ms ago"
    if snap.reconcile_mismatch_count > 0:
        return f"reconciliation mismatches during canary: {snap.reconcile_mismatch_count}"
    if snap.error_count > 5:
        return f"excessive errors during canary soak: {snap.error_count}"
    return None


HEALTH_CHECKERS = {
    "shadow": check_shadow_health,
    "paper": check_paper_health,
    "canary": check_canary_health,
}


# ===========================================================================
# Service management
# ===========================================================================

def restart_service(
    config_dir: Path,
    trade_mode: str,
    canary_mode: bool,
    env_overrides: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    log_fn=print,
) -> bool:
    """Restart the service with stage-specific env overrides.

    Writes a stage-overlay env file that sets PARAPHINA_TRADE_MODE and
    PARAPHINA_CANARY_MODE, then restarts the systemd service.
    """
    overlay_path = config_dir / "stage_overlay.env"
    overlay_lines = [
        f"PARAPHINA_TRADE_MODE={trade_mode}",
    ]
    if canary_mode:
        overlay_lines.append("PARAPHINA_CANARY_MODE=1")
    else:
        overlay_lines.append("PARAPHINA_CANARY_MODE=0")
    if env_overrides:
        for k, v in sorted(env_overrides.items()):
            overlay_lines.append(f"{k}={v}")

    overlay_content = "\n".join(overlay_lines) + "\n"

    if dry_run:
        log_fn(f"[dry-run] Would write {overlay_path}:")
        for line in overlay_lines:
            log_fn(f"  {line}")
        log_fn(f"[dry-run] Would restart {SERVICE_NAME}")
        return True

    overlay_path.write_text(overlay_content)
    log_fn(f"  Wrote stage overlay: {overlay_path}")

    cmd = ["sudo", "systemctl", "restart", SERVICE_NAME]
    log_fn(f"  Restarting service: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log_fn(f"  [error] Restart failed: {result.stderr.strip()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log_fn(f"  [error] Restart timed out")
        return False
    except FileNotFoundError:
        log_fn(f"  [error] systemctl not found")
        return False


def run_config_manager(
    config_dir: Path,
    action: str,
    args: Optional[List[str]] = None,
    log_fn=print,
) -> bool:
    """Run deploy/config_manager.py with the given action."""
    script = Path(__file__).resolve().parent / "config_manager.py"
    cmd = [sys.executable, str(script), "--config-dir", str(config_dir), action]
    if args:
        cmd.extend(args)
    log_fn(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                log_fn(f"    {line}")
        if result.returncode != 0:
            if result.stderr:
                log_fn(f"  [error] {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        log_fn(f"  [error] Config manager failed: {e}")
        return False


# ===========================================================================
# Core deploy pipeline
# ===========================================================================

def build_stages(
    shadow_soak: int,
    paper_soak: int,
    canary_soak: int,
    stop_before: Optional[str] = None,
) -> List[StageConfig]:
    """Build the list of deploy stages."""
    all_stages = [
        StageConfig(
            name="shadow",
            trade_mode="shadow",
            canary_mode=False,
            soak_seconds=shadow_soak,
        ),
        StageConfig(
            name="paper",
            trade_mode="paper",
            canary_mode=False,
            soak_seconds=paper_soak,
        ),
        StageConfig(
            name="canary",
            trade_mode="live",
            canary_mode=True,
            soak_seconds=canary_soak,
            env_overrides={
                "PARAPHINA_LIVE_EXEC_ENABLE": "1",
                "PARAPHINA_LIVE_EXECUTION_CONFIRM": "YES",
            },
        ),
        StageConfig(
            name="live",
            trade_mode="live",
            canary_mode=False,
            soak_seconds=0,  # live is continuous, no soak
            env_overrides={
                "PARAPHINA_LIVE_EXEC_ENABLE": "1",
                "PARAPHINA_LIVE_EXECUTION_CONFIRM": "YES",
            },
        ),
    ]

    if stop_before:
        stop_names = [s.name for s in all_stages]
        if stop_before not in stop_names:
            raise ValueError(
                f"Unknown stop_before stage: '{stop_before}'. "
                f"Valid stages: {stop_names}"
            )
        idx = stop_names.index(stop_before)
        return all_stages[:idx]

    return all_stages


def run_soak(
    stage: StageConfig,
    health_url: str,
    dry_run: bool = False,
    log_fn=print,
) -> Optional[str]:
    """Run a soak period for a stage, polling health at intervals.

    Returns None on success, or a failure reason string.
    """
    if stage.soak_seconds <= 0:
        return None

    if dry_run:
        log_fn(f"[dry-run] Would soak for {stage.soak_seconds}s in {stage.name} mode")
        return None

    log_fn(f"  Soaking for {stage.soak_seconds}s...")
    start = time.time()
    deadline = start + stage.soak_seconds

    checker = HEALTH_CHECKERS.get(stage.name)

    while time.time() < deadline:
        elapsed = time.time() - start
        remaining = deadline - time.time()

        snap = fetch_health_detail(health_url)
        if snap is None:
            # Service might have crashed
            if elapsed > 15:
                return f"cannot reach health endpoint after {elapsed:.0f}s"
            # Give it a moment to come up
            time.sleep(POLL_INTERVAL)
            continue

        # Run stage-specific health check
        if checker:
            failure = checker(snap, elapsed)
            if failure:
                return failure

        # Progress log every 30 seconds
        if int(elapsed) % 30 < POLL_INTERVAL:
            log_fn(
                f"  [{stage.name}] {elapsed:.0f}s/{stage.soak_seconds}s "
                f"ticks={snap.tick_count} errors={snap.error_count} "
                f"healthy={snap.healthy}"
            )

        time.sleep(POLL_INTERVAL)

    # Final health check
    snap = fetch_health_detail(health_url)
    if snap and checker:
        failure = checker(snap, stage.soak_seconds)
        if failure:
            return failure

    return None


def cmd_deploy(args: argparse.Namespace) -> int:
    """Execute the graduated deploy pipeline."""
    config_dir = Path(args.config_dir)
    promoted_env = Path(args.env_file)
    health_url = args.health_url
    dry_run = args.dry_run
    stop_before = getattr(args, "stop_before", None)

    log_fn = print
    start_time = time.time()

    log_fn("=" * 60)
    log_fn("Paraphina Auto-Deploy Pipeline")
    log_fn(f"  Config dir:    {config_dir}")
    log_fn(f"  Promoted env:  {promoted_env}")
    log_fn(f"  Health URL:    {health_url}")
    log_fn(f"  Dry run:       {dry_run}")
    log_fn(f"  Stop before:   {stop_before or '(none)'}")
    log_fn("=" * 60)

    # Validate inputs
    if not promoted_env.exists():
        log_fn(f"[error] Promoted env file not found: {promoted_env}")
        return 1

    if not config_dir.exists() and not dry_run:
        log_fn(f"[error] Config directory does not exist: {config_dir}")
        log_fn("        Run 'config_manager.py init' first.")
        return 1

    # Build stage list
    stages = build_stages(
        shadow_soak=args.shadow_soak,
        paper_soak=args.paper_soak,
        canary_soak=args.canary_soak,
        stop_before=stop_before,
    )
    stage_names = [s.name for s in stages]
    log_fn(f"\nStages: {' -> '.join(stage_names)}")

    # Step 1: Activate the promoted config (symlink rotation)
    log_fn(f"\n[1/3] Activating promoted config...")
    if not dry_run:
        ok = run_config_manager(
            config_dir, "activate",
            [str(promoted_env), "--stage", "shadow"],
            log_fn=log_fn,
        )
        if not ok:
            log_fn("[error] Config activation failed")
            return 1
    else:
        log_fn(f"[dry-run] Would activate {promoted_env}")

    # Step 2: Progress through stages
    stages_passed = []
    for i, stage in enumerate(stages):
        log_fn(f"\n[Stage {i+1}/{len(stages)}] {stage.name.upper()}")
        log_fn(f"  Trade mode: {stage.trade_mode}")
        log_fn(f"  Canary: {stage.canary_mode}")
        log_fn(f"  Soak: {stage.soak_seconds}s")

        # Restart service with stage-specific mode
        if not restart_service(
            config_dir, stage.trade_mode, stage.canary_mode,
            env_overrides=stage.env_overrides,
            dry_run=dry_run, log_fn=log_fn,
        ):
            log_fn(f"\n[FAIL] Service restart failed at stage: {stage.name}")
            _do_rollback(config_dir, f"restart_failed_at_{stage.name}", dry_run, log_fn)
            return 1

        # Wait for service to become healthy
        if not dry_run:
            log_fn(f"  Waiting for service to become healthy...")
            if not wait_for_healthy(health_url, log_fn=log_fn):
                log_fn(f"\n[FAIL] Service did not become healthy at stage: {stage.name}")
                _do_rollback(config_dir, f"startup_timeout_at_{stage.name}", dry_run, log_fn)
                return 1

        # Update stage in state file
        if not dry_run:
            run_config_manager(config_dir, "update-stage", [stage.name], log_fn=log_fn)

        # Run soak period with health monitoring
        failure = run_soak(stage, health_url, dry_run=dry_run, log_fn=log_fn)
        if failure:
            log_fn(f"\n[FAIL] Health gate failed at stage {stage.name}: {failure}")
            _do_rollback(config_dir, f"health_gate_{stage.name}:{failure}", dry_run, log_fn)
            return 1

        stages_passed.append(stage.name)
        log_fn(f"  [PASS] {stage.name} stage complete")

    # Step 3: All stages passed
    elapsed = time.time() - start_time
    log_fn(f"\n{'=' * 60}")
    log_fn(f"[SUCCESS] Deploy complete in {elapsed:.0f}s")
    log_fn(f"  Stages passed: {', '.join(stages_passed)}")
    log_fn(f"  Active config: {promoted_env.name}")
    log_fn(f"{'=' * 60}")

    # Write deploy summary
    if not dry_run:
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": "success",
            "stages_passed": stages_passed,
            "promoted_env": str(promoted_env),
            "duration_seconds": round(elapsed, 1),
        }
        summary_path = config_dir / "last_deploy_summary.json"
        try:
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        except OSError:
            pass

    return 0


def _do_rollback(
    config_dir: Path,
    reason: str,
    dry_run: bool,
    log_fn=print,
) -> bool:
    """Execute rollback to previous config."""
    log_fn(f"\n[ROLLBACK] Rolling back to previous config...")
    log_fn(f"  Reason: {reason}")

    if dry_run:
        log_fn(f"[dry-run] Would rollback and restart")
        return True

    rollback_args = ["--reason", reason]
    ok = run_config_manager(config_dir, "rollback", rollback_args, log_fn=log_fn)
    if not ok:
        log_fn("[error] Rollback failed! Manual intervention required.")
        return False

    log_fn("[ok] Rollback complete.")
    return True


def cmd_rollback(args: argparse.Namespace) -> int:
    """Manual rollback command."""
    config_dir = Path(args.config_dir)
    reason = getattr(args, "reason", "manual rollback via orchestrator")
    dry_run = getattr(args, "dry_run", False)

    if _do_rollback(config_dir, reason, dry_run):
        return 0
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show deploy status including health check."""
    config_dir = Path(args.config_dir)
    health_url = args.health_url

    # Show config state
    run_config_manager(config_dir, "status")

    # Show live health
    print(f"\nLive health ({health_url}/health/detail):")
    snap = fetch_health_detail(health_url)
    if snap:
        print(f"  Healthy:         {snap.healthy}")
        print(f"  Ready:           {snap.ready}")
        print(f"  Uptime:          {snap.uptime_seconds}s")
        print(f"  Tick count:      {snap.tick_count}")
        print(f"  Tick age:        {snap.tick_age_ms}ms")
        print(f"  Errors:          {snap.error_count}")
        print(f"  Recon mismatches: {snap.reconcile_mismatch_count}")
        print(f"  Kill events:     {snap.kill_events_present}")
        print(f"  Trade mode:      {snap.trade_mode}")
        print(f"  Config ID:       {snap.config_id}")
    else:
        print("  [warn] Cannot reach health endpoint")

    # Show last deploy summary
    summary_path = config_dir / "last_deploy_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            print(f"\nLast deploy:")
            print(f"  Result:    {summary.get('result')}")
            print(f"  Timestamp: {summary.get('timestamp')}")
            print(f"  Duration:  {summary.get('duration_seconds')}s")
            print(f"  Stages:    {', '.join(summary.get('stages_passed', []))}")
        except (json.JSONDecodeError, OSError):
            pass

    return 0


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="deploy_orchestrator",
        description="Paraphina graduated auto-deploy pipeline",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help=f"Config directory (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--health-url",
        default=DEFAULT_HEALTH_URL,
        help=f"Base URL for health endpoint (default: {DEFAULT_HEALTH_URL})",
    )
    sub = parser.add_subparsers(dest="command")

    # deploy
    p_deploy = sub.add_parser("deploy", help="Run graduated deploy pipeline")
    p_deploy.add_argument("env_file", help="Path to the promoted .env file")
    p_deploy.add_argument("--shadow-soak", type=int, default=DEFAULT_SHADOW_SOAK,
                          help=f"Shadow soak duration in seconds (default: {DEFAULT_SHADOW_SOAK})")
    p_deploy.add_argument("--paper-soak", type=int, default=DEFAULT_PAPER_SOAK,
                          help=f"Paper soak duration in seconds (default: {DEFAULT_PAPER_SOAK})")
    p_deploy.add_argument("--canary-soak", type=int, default=DEFAULT_CANARY_SOAK,
                          help=f"Canary soak duration in seconds (default: {DEFAULT_CANARY_SOAK})")
    p_deploy.add_argument("--stop-before", choices=["paper", "canary", "live"],
                          help="Stop before this stage (human approves remaining)")
    p_deploy.add_argument("--dry-run", action="store_true",
                          help="Print actions without executing")

    # rollback
    p_rb = sub.add_parser("rollback", help="Rollback to previous config")
    p_rb.add_argument("--reason", default="manual rollback via orchestrator")
    p_rb.add_argument("--dry-run", action="store_true")

    # status
    sub.add_parser("status", help="Show deploy + health status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "deploy": cmd_deploy,
        "rollback": cmd_rollback,
        "status": cmd_status,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
