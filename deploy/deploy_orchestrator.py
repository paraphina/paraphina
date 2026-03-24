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
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# Constants
# ===========================================================================

DEFAULT_CONFIG_DIR = Path("/etc/paraphina")
DEFAULT_HEALTH_URL = "http://127.0.0.1:9898"
SERVICE_NAME = "paraphina_live"
CURRENT_LINK = "current.env"
LEGACY_ENV_FILE = "paraphina_live.env"
STAGE_OVERLAY_FILE = "stage_overlay.env"
LIVE_EXEC_DROPIN_PATH = Path(
    f"/etc/systemd/system/{SERVICE_NAME}.service.d/live_exec_flag.conf"
)
LIVE_EXEC_DROPIN_CONTENT = """[Service]
ExecStartPre=
ExecStartPre=/opt/paraphina/paraphina_live --enable-live-execution --validate-config
ExecStart=
ExecStart=/opt/paraphina/paraphina_live --enable-live-execution
"""

# Default soak durations in seconds
DEFAULT_SHADOW_SOAK = 300   # 5 minutes
DEFAULT_PAPER_SOAK = 600    # 10 minutes
DEFAULT_CANARY_SOAK = 900   # 15 minutes
DEFAULT_PRE_CANARY_POSITION_TOL_BASE = 0.0025
DEFAULT_PRE_CANARY_MAX_OPEN_ORDERS = 0
PRE_CANARY_AUDIT_TIMEOUT = 60
PRE_CANARY_AUDIT_SCRIPT = (
    Path(__file__).resolve().parent.parent / "tools" / "live_venue_audit.py"
)
POST_CANARY_CLEANUP_MAX_PASSES = 3
POST_CANARY_CLEAN_CONFIRM_AUDITS = 2
POST_CANARY_CLEANUP_MIN_SETTLE_SECONDS = 2.0
POST_CANARY_FLATTEN_MIN_SETTLE_SECONDS = 4.0
CANARY_ARTIFACTS_DIR = Path("/var/lib/paraphina/out/canary_artifacts")
CANARY_ARTIFACT_FILES = (
    "telemetry.jsonl",
    "reconcile_drift.jsonl",
    "kill_events.jsonl",
    "summary.json",
    "config_resolved.json",
    "build_info.json",
)
DEFAULT_CANARY_ARTIFACT_RETENTION_BYTES = 5 * 1024 * 1024 * 1024
DEFAULT_CANARY_ARTIFACT_MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024
DEFAULT_CANARY_ARTIFACT_KEEP_DIRS = 6
DEFAULT_CANARY_ARTIFACT_MAX_FILE_BYTES = 256 * 1024 * 1024
DEFAULT_CANARY_ARTIFACT_TAIL_BYTES = 128 * 1024 * 1024
VENUE_NAMES = ["extended", "hyperliquid", "aster", "lighter", "paradex"]
REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_BIN = os.environ.get("CARGO") or shutil.which("cargo") or str(
    (REPO_ROOT.parent / ".cargo" / "bin" / "cargo")
)
LIVE_CLEANUP_BIN = (
    Path(os.environ["PARAPHINA_LIVE_CLEANUP_BIN"]).expanduser()
    if os.environ.get("PARAPHINA_LIVE_CLEANUP_BIN")
    else None
)
if LIVE_CLEANUP_BIN is None:
    for candidate in (
        REPO_ROOT / "target" / "release" / "live_cleanup",
        REPO_ROOT / "target" / "debug" / "live_cleanup",
    ):
        if candidate.exists():
            LIVE_CLEANUP_BIN = candidate
            break
LIVE_CLEANUP_TIMEOUT = 300
LIVE_CLEANUP_FEATURES = ",".join(
    [
        "live",
        "live_hyperliquid",
        "live_lighter",
        "live_extended",
        "live_aster",
        "live_paradex",
    ]
)

# Health polling interval in seconds
POLL_INTERVAL = 10

# Maximum time to wait for service to become healthy after restart
STARTUP_TIMEOUT = 60

# Require sustained health endpoint loss before failing a soak.
# A single timed-out /health/detail poll has proven too brittle during live canary.
HEALTH_UNREACHABLE_GRACE_POLLS = 3

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


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _iter_json_objects(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            while line:
                try:
                    record, idx = decoder.raw_decode(line)
                except json.JSONDecodeError:
                    break
                if isinstance(record, dict):
                    yield record
                line = line[idx:].lstrip()


def _last_execution_segment(path: Path, execution_mode: str) -> List[Dict[str, Any]]:
    last: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    prev_tick: Optional[int] = None
    for record in _iter_json_objects(path):
        if record.get("execution_mode") != execution_mode:
            if current:
                last = current
                current = []
                prev_tick = None
            continue
        tick = _safe_int(record.get("t"))
        if current and tick is not None and prev_tick is not None and tick != prev_tick + 1:
            last = current
            current = []
        current.append(record)
        prev_tick = tick
    if current:
        last = current
    return last


def _record_ts_ms(record: Dict[str, Any]) -> Optional[int]:
    treasury_guidance = record.get("treasury_guidance")
    if isinstance(treasury_guidance, dict):
        as_of_ms = _safe_int(treasury_guidance.get("as_of_ms"))
        if as_of_ms is not None:
            return as_of_ms
    return _safe_int(record.get("last_tick_ms"))


def _iso_utc(ts_ms: Optional[int]) -> Optional[str]:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def load_env_values(paths: List[Path]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            values[key.strip()] = strip_wrapping_quotes(raw_value)
    return values


def env_truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def live_exec_requested(config_dir: Path) -> bool:
    env = load_env_values(
        [
            config_dir / CURRENT_LINK,
            config_dir / LEGACY_ENV_FILE,
            config_dir / STAGE_OVERLAY_FILE,
        ]
    )
    trade_mode = env.get("PARAPHINA_TRADE_MODE", "").strip().lower()
    confirm = env.get("PARAPHINA_LIVE_EXECUTION_CONFIRM", "").strip().upper()
    return (
        trade_mode == "live"
        and env_truthy(env.get("PARAPHINA_LIVE_EXEC_ENABLE"))
        and confirm == "YES"
    )


def sync_live_exec_dropin(config_dir: Path, dry_run: bool = False, log_fn=print) -> bool:
    should_enable = live_exec_requested(config_dir)
    changed = False

    if should_enable:
        if dry_run:
            log_fn(f"[dry-run] Would ensure live-exec drop-in: {LIVE_EXEC_DROPIN_PATH}")
        else:
            LIVE_EXEC_DROPIN_PATH.parent.mkdir(parents=True, exist_ok=True)
            current = (
                LIVE_EXEC_DROPIN_PATH.read_text()
                if LIVE_EXEC_DROPIN_PATH.exists()
                else None
            )
            if current != LIVE_EXEC_DROPIN_CONTENT:
                LIVE_EXEC_DROPIN_PATH.write_text(LIVE_EXEC_DROPIN_CONTENT)
                changed = True
    elif LIVE_EXEC_DROPIN_PATH.exists():
        if dry_run:
            log_fn(f"[dry-run] Would remove stale live-exec drop-in: {LIVE_EXEC_DROPIN_PATH}")
        else:
            LIVE_EXEC_DROPIN_PATH.unlink()
            changed = True

    if dry_run or not changed:
        return True

    cmd = ["sudo", "systemctl", "daemon-reload"]
    log_fn(f"  Reloading systemd: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log_fn(f"  [error] systemd daemon-reload failed: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        log_fn(f"  [error] systemd daemon-reload timed out")
        return False
    except FileNotFoundError:
        log_fn(f"  [error] systemctl not found")
        return False
    return True


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
    overlay_path = config_dir / STAGE_OVERLAY_FILE
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
    if not sync_live_exec_dropin(config_dir, dry_run=dry_run, log_fn=log_fn):
        return False

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


def run_pre_canary_audit(
    config_dir: Path,
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    dry_run: bool = False,
    log_fn=print,
) -> Optional[str]:
    """Audit direct venue-side state before entering live canary."""
    payload, failure = _run_live_venue_audit(
        config_dir,
        position_tol_base=position_tol_base,
        max_open_orders=max_open_orders,
        allow_unknown_open_orders=allow_unknown_open_orders,
        dry_run=dry_run,
        heading="pre-canary venue audit",
        timeout=PRE_CANARY_AUDIT_TIMEOUT,
        log_fn=log_fn,
    )
    if failure:
        return failure
    if payload is not None:
        log_fn("  [PASS] Pre-canary venue audit passed")
    return None


def _run_live_venue_audit(
    config_dir: Path,
    *,
    position_tol_base: float,
    max_open_orders: int,
    allow_unknown_open_orders: bool,
    dry_run: bool,
    heading: str,
    timeout: int,
    log_fn=print,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run the direct venue audit and return (payload, failure_reason)."""
    env_file = config_dir / CURRENT_LINK
    cmd = [
        sys.executable,
        str(PRE_CANARY_AUDIT_SCRIPT),
        "--env-file",
        str(env_file),
        "--position-tol-base",
        str(position_tol_base),
        "--max-open-orders",
        str(max_open_orders),
    ]
    if allow_unknown_open_orders:
        cmd.append("--allow-unknown-open-orders")

    if dry_run:
        log_fn(f"[dry-run] Would run {heading}: {' '.join(cmd)}")
        return None, None

    if not PRE_CANARY_AUDIT_SCRIPT.exists():
        return None, f"audit script not found: {PRE_CANARY_AUDIT_SCRIPT}"
    if not env_file.exists():
        return None, f"active env file missing for audit: {env_file}"

    log_fn(f"  Running {heading}...")
    log_fn(f"    {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"{heading} timed out"
    except OSError as exc:
        return None, f"{heading} failed to start: {exc}"

    payload = None
    stdout = proc.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None

    if isinstance(payload, dict):
        for result in payload.get("results", []):
            venue = result.get("venue", "unknown")
            market = result.get("market", "")
            position_base = float(result.get("position_base", 0.0))
            open_orders = result.get("open_order_count")
            open_orders_known = bool(result.get("open_order_count_known", False))
            open_orders_text = str(open_orders) if open_orders_known else "unknown"
            log_fn(
                "    "
                f"[audit] venue={venue} market={market} "
                f"position_base={position_base:.8f} open_orders={open_orders_text}"
            )
    if proc.returncode == 0:
        return payload, None

    if isinstance(payload, dict):
        violations = payload.get("violations") or []
        if violations:
            preview = "; ".join(str(item) for item in violations[:4])
            if len(violations) > 4:
                preview += f"; ... ({len(violations)} total)"
            return payload, preview

    stderr = proc.stderr.strip()
    if stderr:
        return payload, stderr
    if stdout:
        return payload, stdout
    return payload, f"{heading} failed"


def _run_live_cleanup(
    config_dir: Path,
    *,
    lighter_pos: float,
    extended_pos: float,
    aster_pos: float,
    paradex_pos: float,
    telemetry_path: Optional[Path],
    settle_ms: int,
    dry_run: bool,
    log_fn=print,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    env_file = config_dir / CURRENT_LINK
    cleanup_args = [
        "--env-file",
        str(env_file),
        f"--lighter-pos={lighter_pos}",
        f"--extended-pos={extended_pos}",
        f"--aster-pos={aster_pos}",
        f"--paradex-pos={paradex_pos}",
        f"--settle-ms={settle_ms}",
        "--json-summary",
    ]
    if telemetry_path is not None:
        cleanup_args.extend(["--telemetry-path", str(telemetry_path)])
    if LIVE_CLEANUP_BIN is not None:
        cmd = [str(LIVE_CLEANUP_BIN), *cleanup_args]
    else:
        cmd = [
            CARGO_BIN,
            "run",
            "-p",
            "paraphina",
            "--bin",
            "live_cleanup",
            "--features",
            LIVE_CLEANUP_FEATURES,
            "--",
            *cleanup_args,
        ]

    if dry_run:
        log_fn(f"[dry-run] Would run stop-before-live cleanup: {' '.join(cmd)}")
        return None, None

    log_fn("  Running stop-before-live cleanup...")
    log_fn(f"    {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=LIVE_CLEANUP_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None, "stop-before-live cleanup timed out"
    except OSError as exc:
        return None, f"stop-before-live cleanup failed to start: {exc}"

    cleanup_summary = None
    stdout = proc.stdout.strip()
    if stdout:
        try:
            cleanup_summary = json.loads(stdout)
            log_fn(
                "    "
                f"cleanup_summary result={cleanup_summary.get('result')} "
                f"venues={cleanup_summary.get('venues_touched', [])} "
                f"est_cost_usd={cleanup_summary.get('total_estimated_cleanup_cost_usd')}"
            )
        except json.JSONDecodeError:
            cleanup_summary = None
            for line in stdout.splitlines():
                log_fn(f"    {line}")
    if proc.returncode == 0:
        if cleanup_summary is None:
            return None, "stop-before-live cleanup produced no parseable JSON summary"
        return cleanup_summary, None
    stderr = proc.stderr.strip()
    if stderr:
        return cleanup_summary, stderr
    if stdout:
        return cleanup_summary, stdout
    return cleanup_summary, "stop-before-live cleanup failed"


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _artifact_retention_bytes() -> int:
    return (
        _parse_positive_int(os.environ.get("PARAPHINA_CANARY_ARTIFACT_RETENTION_BYTES"))
        or DEFAULT_CANARY_ARTIFACT_RETENTION_BYTES
    )


def _artifact_min_free_bytes() -> int:
    return (
        _parse_positive_int(os.environ.get("PARAPHINA_CANARY_ARTIFACT_MIN_FREE_BYTES"))
        or DEFAULT_CANARY_ARTIFACT_MIN_FREE_BYTES
    )


def _artifact_keep_dirs() -> int:
    return (
        _parse_positive_int(os.environ.get("PARAPHINA_CANARY_ARTIFACT_KEEP_DIRS"))
        or DEFAULT_CANARY_ARTIFACT_KEEP_DIRS
    )


def _artifact_max_file_bytes() -> int:
    return (
        _parse_positive_int(os.environ.get("PARAPHINA_CANARY_ARTIFACT_MAX_FILE_BYTES"))
        or DEFAULT_CANARY_ARTIFACT_MAX_FILE_BYTES
    )


def _artifact_tail_bytes() -> int:
    return (
        _parse_positive_int(os.environ.get("PARAPHINA_CANARY_ARTIFACT_TAIL_BYTES"))
        or DEFAULT_CANARY_ARTIFACT_TAIL_BYTES
    )


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for child in path.rglob("*"):
        try:
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def _prune_canary_artifacts_for_space(*, log_fn=print) -> List[str]:
    if not CANARY_ARTIFACTS_DIR.exists():
        return []

    try:
        artifact_dirs = sorted(
            [path for path in CANARY_ARTIFACTS_DIR.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
    except OSError as exc:
        log_fn(f"  [warn] Failed to enumerate canary artifacts for pruning: {exc}")
        return []

    if not artifact_dirs:
        return []

    retention_bytes = _artifact_retention_bytes()
    min_free_bytes = _artifact_min_free_bytes()
    keep_dirs = max(1, _artifact_keep_dirs())
    total_bytes = sum(_dir_size_bytes(path) for path in artifact_dirs)
    removed: List[str] = []
    remaining = list(artifact_dirs)

    while remaining:
        try:
            free_bytes = shutil.disk_usage(CANARY_ARTIFACTS_DIR.parent).free
        except OSError as exc:
            log_fn(f"  [warn] Failed to read disk usage for artifact pruning: {exc}")
            break

        over_retention = total_bytes > retention_bytes
        low_free = free_bytes < min_free_bytes
        if not over_retention and not low_free:
            break
        if len(remaining) <= 1:
            break

        if low_free:
            # Emergency mode: once the host is already below the free-space floor,
            # keeping historical artifacts matters less than preserving the
            # ability to return the box to shadow and write cleanup metadata.
            pass
        elif len(remaining) <= keep_dirs:
            break

        candidate = remaining.pop(0)
        candidate_size = _dir_size_bytes(candidate)
        try:
            shutil.rmtree(candidate)
        except OSError as exc:
            log_fn(f"  [warn] Failed to prune canary artifact {candidate}: {exc}")
            continue

        total_bytes = max(0, total_bytes - candidate_size)
        removed.append(str(candidate))
        log_fn(
            "  "
            f"Pruned canary artifact {candidate.name} "
            f"({candidate_size / (1024 * 1024):.1f} MiB)"
        )

    return removed


def _copy_jsonl_tail(src: Path, dst: Path, tail_bytes: int) -> int:
    size_bytes = src.stat().st_size
    start_offset = max(0, size_bytes - max(1, tail_bytes))
    copied_bytes = 0
    with src.open("rb") as in_fh, dst.open("wb") as out_fh:
        if start_offset > 0:
            in_fh.seek(start_offset)
            _ = in_fh.readline()
        while True:
            chunk = in_fh.read(1024 * 1024)
            if not chunk:
                break
            out_fh.write(chunk)
            copied_bytes += len(chunk)
    return copied_bytes


def _snapshot_artifact_file(src: Path, dst: Path, *, log_fn=print) -> Dict[str, Any]:
    size_bytes = src.stat().st_size
    max_file_bytes = _artifact_max_file_bytes()
    tail_bytes = _artifact_tail_bytes()

    if src.suffix == ".jsonl" and size_bytes > max_file_bytes:
        copied_bytes = _copy_jsonl_tail(src, dst, tail_bytes)
        log_fn(
            "  "
            f"Snapshotted tail of {src.name}: "
            f"{copied_bytes / (1024 * 1024):.1f} MiB "
            f"from original {size_bytes / (1024 * 1024):.1f} MiB"
        )
        return {
            "name": src.name,
            "mode": "tail_jsonl",
            "source_size_bytes": size_bytes,
            "copied_size_bytes": copied_bytes,
            "tail_bytes": tail_bytes,
        }

    shutil.copy2(src, dst)
    return {
        "name": src.name,
        "mode": "full",
        "source_size_bytes": size_bytes,
        "copied_size_bytes": size_bytes,
    }


def _cleanup_settle_seconds(
    config_dir: Path,
    *,
    has_position_cleanup: bool,
) -> float:
    env = load_env_values(
        [
            config_dir / CURRENT_LINK,
            config_dir / STAGE_OVERLAY_FILE,
        ]
    )
    poll_candidates: List[int] = []
    for key, raw in env.items():
        if key == "PARAPHINA_LIVE_ACCOUNT_POLL_MS" or key.endswith("_ACCOUNT_POLL_MS"):
            parsed = _parse_positive_int(raw)
            if parsed is not None:
                poll_candidates.append(parsed)
    max_poll_ms = max(poll_candidates) if poll_candidates else 1000
    base = (
        POST_CANARY_FLATTEN_MIN_SETTLE_SECONDS
        if has_position_cleanup
        else POST_CANARY_CLEANUP_MIN_SETTLE_SECONDS
    )
    return max(base, (2 * max_poll_ms) / 1000.0 + 0.5)


def _audit_cleanup_state(
    payload: Optional[Dict[str, Any]],
    *,
    position_tol_base: float,
) -> Tuple[bool, Dict[str, float], List[str], List[Dict[str, Any]], bool]:
    cleanup_positions = {
        "lighter": 0.0,
        "extended": 0.0,
        "aster": 0.0,
        "paradex": 0.0,
    }
    unsupported_residuals: List[str] = []
    dirty_results: List[Dict[str, Any]] = []
    has_position_cleanup = False

    if not isinstance(payload, dict):
        return False, cleanup_positions, unsupported_residuals, dirty_results, has_position_cleanup

    for result in payload.get("results", []):
        venue = str(result.get("venue", ""))
        position_base = float(result.get("position_base", 0.0))
        open_order_known = bool(result.get("open_order_count_known", False))
        open_order_count = int(result.get("open_order_count", 0) or 0)
        needs_position_cleanup = abs(position_base) > position_tol_base
        needs_order_cleanup = open_order_known and open_order_count > 0
        if venue in cleanup_positions:
            cleanup_positions[venue] = position_base
        if not (needs_position_cleanup or needs_order_cleanup):
            continue
        dirty_results.append(
            {
                "venue": venue,
                "position_base": position_base,
                "open_order_count": open_order_count,
                "open_order_count_known": open_order_known,
            }
        )
        has_position_cleanup = has_position_cleanup or needs_position_cleanup
        if venue == "hyperliquid":
            unsupported_residuals.append(
                f"hyperliquid residual position={position_base:.8f} open_orders={open_order_count}"
            )

    return (
        len(dirty_results) == 0,
        cleanup_positions,
        unsupported_residuals,
        dirty_results,
        has_position_cleanup,
    )


def _write_cleanup_report(
    artifact_dir: Optional[Path],
    report: Dict[str, Any],
    *,
    log_fn=print,
) -> None:
    if artifact_dir is None:
        return
    try:
        (artifact_dir / "cleanup_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    except OSError as exc:
        log_fn(f"  [warn] Failed to write cleanup report: {exc}")
    _write_session_pnl_summary(artifact_dir, report, log_fn=log_fn)


def _build_session_pnl_summary(
    artifact_dir: Path,
    cleanup_report: Dict[str, Any],
) -> Dict[str, Any]:
    telemetry_path = artifact_dir / "telemetry.jsonl"
    base_summary: Dict[str, Any] = {
        "schema_version": 2,
        "artifact_dir": str(artifact_dir),
        "cleanup_result": cleanup_report.get("result"),
        "cleanup_completed_at": cleanup_report.get("completed_at"),
    }
    passes = cleanup_report.get("passes") or []
    if passes:
        first_pass = passes[0]
        base_summary["pre_cleanup_dirty_results"] = first_pass.get("dirty_results", [])
        base_summary["pre_cleanup_residual_positions_base"] = first_pass.get(
            "cleanup_positions", {}
        )
        last_pass = passes[-1]
        audit_payload = last_pass.get("audit_payload")
        if isinstance(audit_payload, dict):
            base_summary["post_cleanup_flat"] = bool(audit_payload.get("ok"))
    else:
        base_summary["pre_cleanup_dirty_results"] = []
        base_summary["pre_cleanup_residual_positions_base"] = {}

    cleanup_execution_summaries = []
    cleanup_estimated_pnl_impact = 0.0
    cleanup_venues_touched: List[str] = []
    cleanup_action_count = 0
    for item in passes:
        summary = item.get("cleanup_execution_summary")
        if not isinstance(summary, dict):
            continue
        cleanup_execution_summaries.append(summary)
        cleanup_estimated_pnl_impact += (
            _safe_float(summary.get("total_estimated_cleanup_cost_usd")) or 0.0
        )
        cleanup_action_count += len(summary.get("actions") or [])
        for venue in summary.get("venues_touched") or []:
            venue_name = str(venue)
            if venue_name not in cleanup_venues_touched:
                cleanup_venues_touched.append(venue_name)
    if cleanup_execution_summaries:
        base_summary["cleanup_cost_confidence"] = "estimated"
    else:
        base_summary["cleanup_cost_confidence"] = None
    base_summary["cleanup_estimated_pnl_impact_usd"] = cleanup_estimated_pnl_impact
    base_summary["cleanup_venues_touched"] = cleanup_venues_touched
    base_summary["cleanup_action_count"] = cleanup_action_count

    if not telemetry_path.exists():
        base_summary["session_pnl_status"] = "telemetry_missing"
        return base_summary

    segment = _last_execution_segment(telemetry_path, "live")
    if not segment:
        base_summary["session_pnl_status"] = "no_live_segment"
        return base_summary

    pnl_values: List[float] = []
    fills_by_venue = {venue: 0 for venue in VENUE_NAMES}
    fill_base_by_venue = {venue: 0.0 for venue in VENUE_NAMES}
    startup_pnl_baseline = None
    for record in segment:
        pnl_total = _safe_float(record.get("pnl_total"))
        if pnl_total is not None:
            pnl_values.append(pnl_total)
        if isinstance(record.get("startup_pnl_baseline"), dict):
            startup_pnl_baseline = record.get("startup_pnl_baseline")
        fills = record.get("fills")
        if not isinstance(fills, list):
            continue
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            venue_index = _safe_int(fill.get("venue_index"))
            if venue_index is None or not (0 <= venue_index < len(VENUE_NAMES)):
                continue
            venue = VENUE_NAMES[venue_index]
            fills_by_venue[venue] += 1
            size = _safe_float(fill.get("size"))
            if size is not None:
                fill_base_by_venue[venue] += size

    final_record = segment[-1]
    residual_positions = final_record.get("venue_position_tao")
    final_live_positions = {}
    if isinstance(residual_positions, list):
        for idx, venue in enumerate(VENUE_NAMES):
            if idx < len(residual_positions):
                final_live_positions[venue] = residual_positions[idx]

    pre_cleanup_residual_positions = base_summary.get(
        "pre_cleanup_residual_positions_base", {}
    )
    had_cleanup_residual = any(
        abs(_safe_float(value) or 0.0) > 0.0
        for value in pre_cleanup_residual_positions.values()
    )
    cleanup_ok = bool(base_summary.get("post_cleanup_flat")) and cleanup_report.get(
        "result"
    ) == "success"
    final_live_mark_pnl_total = _safe_float(final_record.get("pnl_total"))
    session_pnl_after_cleanup_estimated = (
        final_live_mark_pnl_total - cleanup_estimated_pnl_impact
        if final_live_mark_pnl_total is not None
        else None
    )
    if cleanup_ok and had_cleanup_residual and cleanup_execution_summaries:
        session_pnl_status = "session_pnl_after_cleanup_estimated"
    elif cleanup_ok and had_cleanup_residual:
        session_pnl_status = "final_live_mark_with_residual_inventory_then_flattened"
    elif cleanup_ok:
        session_pnl_status = "final_live_mark_flat"
    else:
        session_pnl_status = "final_live_mark_cleanup_incomplete"

    base_summary.update(
        {
            "session_pnl_status": session_pnl_status,
            "live_segment": {
                "execution_mode": "live",
                "tick_count": len(segment),
                "start_tick": _safe_int(segment[0].get("t")),
                "end_tick": _safe_int(final_record.get("t")),
                "start_utc": _iso_utc(_record_ts_ms(segment[0])),
                "end_utc": _iso_utc(_record_ts_ms(final_record)),
                "startup_pnl_baseline": startup_pnl_baseline,
                "initial_pnl_total_usd": pnl_values[0] if pnl_values else None,
                "peak_pnl_total_usd": max(pnl_values) if pnl_values else None,
                "trough_pnl_total_usd": min(pnl_values) if pnl_values else None,
                "final_live_mark_pnl_total_usd": final_live_mark_pnl_total,
                "final_live_mark_pnl_realised_usd": _safe_float(
                    final_record.get("pnl_realised")
                ),
                "final_live_mark_pnl_unrealised_usd": _safe_float(
                    final_record.get("pnl_unrealised")
                ),
                # Backward-compatible aliases for downstream consumers.
                "final_live_pnl_total_usd": final_live_mark_pnl_total,
                "final_live_pnl_realised_usd": _safe_float(
                    final_record.get("pnl_realised")
                ),
                "final_live_pnl_unrealised_usd": _safe_float(
                    final_record.get("pnl_unrealised")
                ),
                "final_live_q_global_base": _safe_float(
                    final_record.get("q_global_tao")
                ),
                "final_live_q_gross_base": _safe_float(
                    final_record.get("q_gross_tao")
                ),
                "final_live_q_max_abs_venue_base": _safe_float(
                    final_record.get("q_max_abs_venue_tao")
                ),
                "final_live_positions_base": final_live_positions,
                "fills_total": sum(fills_by_venue.values()),
                "fills_by_venue": fills_by_venue,
                "fill_base_by_venue": fill_base_by_venue,
                "healthy_venues_used_count": _safe_int(
                    final_record.get("healthy_venues_used_count")
                ),
                "would_send_orders_count": _safe_int(
                    final_record.get("would_send_orders_count")
                ),
            },
            "session_pnl_after_cleanup_estimated_usd": session_pnl_after_cleanup_estimated,
        }
    )
    return base_summary


def _write_session_pnl_summary(
    artifact_dir: Optional[Path],
    cleanup_report: Dict[str, Any],
    *,
    log_fn=print,
) -> None:
    if artifact_dir is None:
        return
    try:
        summary = _build_session_pnl_summary(artifact_dir, cleanup_report)
        (artifact_dir / "session_pnl_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    except OSError as exc:
        log_fn(f"  [warn] Failed to write session PnL summary: {exc}")
    except json.JSONDecodeError as exc:
        log_fn(f"  [warn] Failed to parse telemetry while writing session PnL summary: {exc}")


def _snapshot_stop_before_live_artifacts(
    config_dir: Path,
    promoted_env: Path,
    health_url: str,
    *,
    dry_run: bool,
    log_fn=print,
) -> Optional[Path]:
    capture_ts = datetime.now(timezone.utc)
    destination = CANARY_ARTIFACTS_DIR / (
        f"{capture_ts.strftime('%Y%m%dT%H%M%SZ')}_{promoted_env.stem}"
    )
    if dry_run:
        log_fn(f"[dry-run] Would snapshot canary artifacts to {destination}")
        return destination

    _prune_canary_artifacts_for_space(log_fn=log_fn)
    destination.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    copied_details: List[Dict[str, Any]] = []
    out_dir = Path("/var/lib/paraphina/out")
    for name in CANARY_ARTIFACT_FILES:
        src = out_dir / name
        if not src.exists():
            continue
        try:
            copied_details.append(
                _snapshot_artifact_file(src, destination / name, log_fn=log_fn)
            )
            copied.append(name)
        except OSError as exc:
            log_fn(f"  [warn] Failed to snapshot {name}: {exc}")

    env_file = config_dir / CURRENT_LINK
    if env_file.exists():
        try:
            shutil.copy2(env_file, destination / env_file.name)
            copied.append(env_file.name)
        except OSError as exc:
            log_fn(f"  [warn] Failed to snapshot {env_file.name}: {exc}")

    stage_overlay = config_dir / STAGE_OVERLAY_FILE
    if stage_overlay.exists():
        try:
            shutil.copy2(stage_overlay, destination / stage_overlay.name)
            copied.append(stage_overlay.name)
        except OSError as exc:
            log_fn(f"  [warn] Failed to snapshot {stage_overlay.name}: {exc}")

    if promoted_env.exists():
        try:
            shutil.copy2(promoted_env, destination / promoted_env.name)
            copied.append(promoted_env.name)
        except OSError as exc:
            log_fn(f"  [warn] Failed to snapshot {promoted_env.name}: {exc}")

    health_snapshot = fetch_health_detail(health_url)
    metadata = {
        "captured_at": capture_ts.isoformat(),
        "promoted_env": str(promoted_env),
        "active_env": str(env_file),
        "health_detail": asdict(health_snapshot) if health_snapshot else None,
        "copied_files": copied,
        "copied_file_details": copied_details,
    }
    try:
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except OSError as exc:
        log_fn(f"  [warn] Failed to write artifact metadata: {exc}")
    log_fn(f"  Snapshotted canary artifacts: {destination}")
    return destination


def _finalize_stop_before_live(
    config_dir: Path,
    promoted_env: Path,
    health_url: str,
    *,
    position_tol_base: float,
    allow_unknown_open_orders: bool,
    dry_run: bool,
    log_fn=print,
) -> Optional[str]:
    """Return the host to shadow and clean venue state after a stop-before-live canary."""
    log_fn("\n[POST-CANARY] Returning to shadow baseline and cleaning venue state...")
    artifact_dir: Optional[Path] = None
    cleanup_report: Dict[str, Any] = {
        "schema_version": 2,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "position_tol_base": position_tol_base,
        "passes": [],
    }
    try:
        artifact_dir = _snapshot_stop_before_live_artifacts(
            config_dir,
            promoted_env,
            health_url,
            dry_run=dry_run,
            log_fn=log_fn,
        )
    except OSError as exc:
        log_fn(f"  [warn] Failed to snapshot canary artifacts: {exc}")

    if not dry_run:
        ok = run_config_manager(
            config_dir,
            "activate",
            [str(promoted_env), "--stage", "shadow"],
            log_fn=log_fn,
        )
        if not ok:
            return "failed to activate promoted env into shadow after stop-before-live canary"
    else:
        log_fn(f"[dry-run] Would activate {promoted_env} into shadow")

    if not restart_service(
        config_dir,
        "shadow",
        False,
        dry_run=dry_run,
        log_fn=log_fn,
    ):
        return "failed to restart shadow baseline after stop-before-live canary"

    if not dry_run:
        log_fn("  Waiting for shadow baseline to become healthy...")
        if not wait_for_healthy(health_url, log_fn=log_fn):
            return "shadow baseline did not become healthy after stop-before-live canary"
        run_config_manager(config_dir, "update-stage", ["shadow"], log_fn=log_fn)

    current_heading = "post-canary venue audit"
    audit_payload, audit_failure = _run_live_venue_audit(
        config_dir,
        position_tol_base=position_tol_base,
        max_open_orders=0,
        allow_unknown_open_orders=allow_unknown_open_orders,
        dry_run=dry_run,
        heading=current_heading,
        timeout=PRE_CANARY_AUDIT_TIMEOUT,
        log_fn=log_fn,
    )
    if audit_failure and audit_payload is None:
        cleanup_report["result"] = "audit_failed_to_run"
        cleanup_report["failure"] = audit_failure
        _write_cleanup_report(artifact_dir, cleanup_report, log_fn=log_fn)
        return f"{current_heading} failed to run: {audit_failure}"

    for pass_idx in range(1, POST_CANARY_CLEANUP_MAX_PASSES + 1):
        is_clean, cleanup_positions, unsupported_residuals, dirty_results, has_position_cleanup = _audit_cleanup_state(
            audit_payload,
            position_tol_base=position_tol_base,
        )
        cleanup_report["passes"].append(
            {
                "pass": pass_idx,
                "heading": current_heading,
                "audit_failure": audit_failure,
                "audit_payload": audit_payload,
                "dirty_results": dirty_results,
                "cleanup_positions": cleanup_positions,
                "cleanup_execution_summary": None,
            }
        )

        if unsupported_residuals:
            cleanup_report["result"] = "unsupported_residuals"
            cleanup_report["failure"] = unsupported_residuals
            _write_cleanup_report(artifact_dir, cleanup_report, log_fn=log_fn)
            return "; ".join(unsupported_residuals)

        if is_clean:
            confirmed = True
            for confirm_idx in range(1, POST_CANARY_CLEAN_CONFIRM_AUDITS):
                settle_seconds = _cleanup_settle_seconds(
                    config_dir,
                    has_position_cleanup=False,
                )
                if not dry_run:
                    log_fn(
                        f"  Waiting {settle_seconds:.1f}s for clean-state confirmation "
                        f"({confirm_idx + 1}/{POST_CANARY_CLEAN_CONFIRM_AUDITS})..."
                    )
                    time.sleep(settle_seconds)
                confirm_heading = f"post-cleanup confirmation audit {confirm_idx + 1}"
                confirm_payload, confirm_failure = _run_live_venue_audit(
                    config_dir,
                    position_tol_base=position_tol_base,
                    max_open_orders=0,
                    allow_unknown_open_orders=allow_unknown_open_orders,
                    dry_run=dry_run,
                    heading=confirm_heading,
                    timeout=PRE_CANARY_AUDIT_TIMEOUT,
                    log_fn=log_fn,
                )
                confirm_clean, _, confirm_unsupported, confirm_dirty, _ = _audit_cleanup_state(
                    confirm_payload,
                    position_tol_base=position_tol_base,
                )
                cleanup_report["passes"].append(
                    {
                        "pass": f"{pass_idx}.confirm{confirm_idx + 1}",
                        "heading": confirm_heading,
                        "audit_failure": confirm_failure,
                        "audit_payload": confirm_payload,
                        "dirty_results": confirm_dirty,
                    }
                )
                if confirm_failure or confirm_unsupported or not confirm_clean:
                    confirmed = False
                    audit_payload = confirm_payload
                    audit_failure = confirm_failure
                    current_heading = confirm_heading
                    break

            if confirmed:
                cleanup_report["result"] = "success"
                cleanup_report["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_cleanup_report(artifact_dir, cleanup_report, log_fn=log_fn)
                return None

        if pass_idx == POST_CANARY_CLEANUP_MAX_PASSES:
            break

        cleanup_summary, cleanup_failure = _run_live_cleanup(
            config_dir,
            lighter_pos=cleanup_positions["lighter"],
            extended_pos=cleanup_positions["extended"],
            aster_pos=cleanup_positions["aster"],
            paradex_pos=cleanup_positions["paradex"],
            telemetry_path=(
                artifact_dir / "telemetry.jsonl"
                if artifact_dir is not None and (artifact_dir / "telemetry.jsonl").exists()
                else None
            ),
            settle_ms=int(_cleanup_settle_seconds(
                config_dir,
                has_position_cleanup=has_position_cleanup,
            ) * 1000),
            dry_run=dry_run,
            log_fn=log_fn,
        )
        cleanup_report["passes"][-1]["cleanup_execution_summary"] = cleanup_summary
        cleanup_report["passes"][-1]["cleanup_failure"] = cleanup_failure
        if cleanup_failure:
            cleanup_report["result"] = "cleanup_failed"
            cleanup_report["failure"] = cleanup_failure
            _write_cleanup_report(artifact_dir, cleanup_report, log_fn=log_fn)
            return cleanup_failure

        settle_seconds = _cleanup_settle_seconds(
            config_dir,
            has_position_cleanup=has_position_cleanup,
        )
        if not dry_run:
            log_fn(f"  Waiting {settle_seconds:.1f}s for venue/account convergence...")
            time.sleep(settle_seconds)

        current_heading = f"post-cleanup venue audit pass {pass_idx}"
        audit_payload, audit_failure = _run_live_venue_audit(
            config_dir,
            position_tol_base=position_tol_base,
            max_open_orders=0,
            allow_unknown_open_orders=allow_unknown_open_orders,
            dry_run=dry_run,
            heading=current_heading,
            timeout=PRE_CANARY_AUDIT_TIMEOUT,
            log_fn=log_fn,
        )
        if audit_failure and audit_payload is None:
            cleanup_report["result"] = "audit_failed_to_run"
            cleanup_report["failure"] = audit_failure
            _write_cleanup_report(artifact_dir, cleanup_report, log_fn=log_fn)
            return f"{current_heading} failed to run: {audit_failure}"

    cleanup_report["result"] = "not_converged"
    cleanup_report["completed_at"] = datetime.now(timezone.utc).isoformat()
    cleanup_report["failure"] = audit_failure
    _write_cleanup_report(artifact_dir, cleanup_report, log_fn=log_fn)
    if audit_failure:
        return f"{current_heading} failed: {audit_failure}"
    return "cleanup did not converge within bounded retries"


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
    consecutive_unreachable = 0
    first_unreachable_ts: Optional[float] = None

    checker = HEALTH_CHECKERS.get(stage.name)

    while time.time() < deadline:
        elapsed = time.time() - start
        remaining = deadline - time.time()

        snap = fetch_health_detail(health_url)
        if snap is None:
            consecutive_unreachable += 1
            if first_unreachable_ts is None:
                first_unreachable_ts = time.time()
            # Service might have crashed, but require sustained loss before rollback.
            if elapsed > 15 and consecutive_unreachable >= HEALTH_UNREACHABLE_GRACE_POLLS:
                unreachable_for = time.time() - first_unreachable_ts
                return (
                    "cannot reach health endpoint "
                    f"for {unreachable_for:.0f}s "
                    f"({consecutive_unreachable} consecutive polls, "
                    f"stage_elapsed={elapsed:.0f}s)"
                )
            # Give it a moment to come up
            time.sleep(POLL_INTERVAL)
            continue
        consecutive_unreachable = 0
        first_unreachable_ts = None

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

        if stage.name == "canary" and not args.skip_pre_canary_audit:
            audit_failure = run_pre_canary_audit(
                config_dir,
                position_tol_base=args.pre_canary_position_tol_base,
                max_open_orders=args.pre_canary_max_open_orders,
                allow_unknown_open_orders=args.allow_unknown_open_orders,
                dry_run=dry_run,
                log_fn=log_fn,
            )
            if audit_failure:
                log_fn(f"\n[FAIL] Pre-canary venue audit failed: {audit_failure}")
                _do_rollback(
                    config_dir,
                    f"pre_canary_audit_failed:{audit_failure}",
                    dry_run,
                    log_fn,
                )
                return 1

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
    if stop_before == "live":
        finalize_failure = _finalize_stop_before_live(
            config_dir,
            promoted_env,
            health_url,
            position_tol_base=args.pre_canary_position_tol_base,
            allow_unknown_open_orders=args.allow_unknown_open_orders,
            dry_run=dry_run,
            log_fn=log_fn,
        )
        if finalize_failure:
            log_fn(f"\n[FAIL] Stop-before-live cleanup failed: {finalize_failure}")
            return 1

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
    p_deploy.add_argument("--skip-pre-canary-audit", action="store_true",
                          help="Skip the direct venue-side audit before canary.")
    p_deploy.add_argument("--pre-canary-position-tol-base", type=float,
                          default=DEFAULT_PRE_CANARY_POSITION_TOL_BASE,
                          help=("Maximum allowed absolute per-venue base position before canary "
                                f"(default: {DEFAULT_PRE_CANARY_POSITION_TOL_BASE})"))
    p_deploy.add_argument("--pre-canary-max-open-orders", type=int,
                          default=DEFAULT_PRE_CANARY_MAX_OPEN_ORDERS,
                          help=("Maximum allowed direct venue-side open orders per venue before canary "
                                f"(default: {DEFAULT_PRE_CANARY_MAX_OPEN_ORDERS})"))
    p_deploy.add_argument("--allow-unknown-open-orders", action="store_true",
                          help="Allow venues whose direct open-order count cannot be determined.")
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
