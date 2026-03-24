#!/usr/bin/env python3
"""
config_manager.py — Config lifecycle manager for Paraphina auto-deploy.

Manages symlink rotation, state tracking, and rollback for promoted configs.

Usage:
    # Activate a promoted .env file (rotates symlinks, validates config)
    python3 deploy/config_manager.py activate <path-to-promoted.env> [--config-dir /etc/paraphina]

    # Rollback to the previous known-good config
    python3 deploy/config_manager.py rollback [--config-dir /etc/paraphina]

    # Show current deploy state
    python3 deploy/config_manager.py status [--config-dir /etc/paraphina]

    # Initialise the config directory structure (first-time setup)
    python3 deploy/config_manager.py init --base-env <path> [--config-dir /etc/paraphina]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
CURRENT_LINK = "current.env"
PREVIOUS_LINK = "previous.env"
STATE_FILE = "deploy_state.json"
HISTORY_FILE = "deploy_history.jsonl"
SERVICE_NAME = "paraphina_live"
STAGE_OVERLAY_FILE = "stage_overlay.env"
LEGACY_ENV_FILE = "paraphina_live.env"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path("/opt/paraphina")
RUNTIME_CONFIGS_DIR = RUNTIME_ROOT / "configs"
CANARY_SNAPSHOT_DIR = "canary_profiles"
LIVE_EXEC_DROPIN_PATH = Path(
    f"/etc/systemd/system/{SERVICE_NAME}.service.d/live_exec_flag.conf"
)
LIVE_EXEC_DROPIN_CONTENT = """[Service]
ExecStartPre=
ExecStartPre=/opt/paraphina/paraphina_live --enable-live-execution --validate-config
ExecStart=
ExecStart=/opt/paraphina/paraphina_live --enable-live-execution
"""


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class DeployRecord:
    """A single deploy/rollback event."""
    timestamp: str
    action: str  # "activate" | "rollback"
    config_file: str  # concrete filename (not symlink)
    config_sha256: str
    previous_config: Optional[str] = None
    reason: Optional[str] = None
    git_commit: Optional[str] = None
    stage: Optional[str] = None  # deploy stage when activated (shadow/paper/canary/live)


@dataclass
class DeployState:
    """Current state of the deploy lifecycle."""
    schema_version: int = 1
    active_config: Optional[str] = None
    active_config_sha256: Optional[str] = None
    previous_config: Optional[str] = None
    previous_config_sha256: Optional[str] = None
    deploy_timestamp: Optional[str] = None
    current_stage: Optional[str] = None
    rollback_count: int = 0
    last_rollback_timestamp: Optional[str] = None
    last_rollback_reason: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeployState":
        d.pop("schema_version", None)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ===========================================================================
# Utilities
# ===========================================================================

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def shorten_snapshot_component(raw: str, max_len: int) -> str:
    """Bound snapshot filename components so nested promotions stay under FS limits."""
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    keep = max_len - len(digest) - 1
    if keep <= 0:
        return digest[:max_len]
    return f"{raw[:keep]}_{digest}"


def git_commit_short() -> Optional[str]:
    """Return short git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(config_dir: Path) -> DeployState:
    """Load deploy state from disk, or return default."""
    state_path = config_dir / STATE_FILE
    if state_path.exists():
        try:
            with open(state_path) as f:
                return DeployState.from_dict(json.load(f))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    return DeployState()


def save_state(config_dir: Path, state: DeployState) -> None:
    """Atomically write deploy state."""
    state_path = config_dir / STATE_FILE
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
        f.write("\n")
    tmp_path.rename(state_path)


def append_history(config_dir: Path, record: DeployRecord) -> None:
    """Append a deploy event to the history JSONL."""
    history_path = config_dir / HISTORY_FILE
    with open(history_path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def resolve_symlink_target(config_dir: Path, link_name: str) -> Optional[str]:
    """Resolve a symlink to its target filename."""
    link = config_dir / link_name
    if link.is_symlink():
        target = os.readlink(link)
        return os.path.basename(target)
    return None


def validate_env_file(path: Path) -> bool:
    """Basic validation: file exists, non-empty, looks like a shell env file."""
    if not path.exists():
        return False
    content = path.read_text()
    if not content.strip():
        return False
    # Every non-empty, non-comment line should be KEY=VALUE or export KEY=VALUE
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Remove optional 'export ' prefix
        if stripped.startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            return False
    return True


def sync_runtime_configs(dry_run: bool = False) -> bool:
    """Copy repo-managed config files into the runtime tree under /opt/paraphina."""
    source_dir = PROJECT_ROOT / "configs"
    if not source_dir.exists():
        print(f"[error] Runtime config source missing: {source_dir}", file=sys.stderr)
        return False

    if dry_run:
        print(f"[dry-run] Would sync runtime configs: {source_dir} -> {RUNTIME_CONFIGS_DIR}")
        return True

    file_count = 0
    try:
        for src in source_dir.rglob("*"):
            rel = src.relative_to(source_dir)
            dest = RUNTIME_CONFIGS_DIR / rel
            if src.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            file_count += 1
    except OSError as err:
        print(
            f"[error] Failed to sync runtime configs into {RUNTIME_CONFIGS_DIR}: {err}",
            file=sys.stderr,
        )
        return False

    print(f"[ok] Synced runtime configs: {file_count} files -> {RUNTIME_CONFIGS_DIR}")
    return True


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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


def resolve_canary_profile_source(raw_value: str, source_env: Path) -> Path:
    """Resolve a canary profile reference from an env file to a concrete source file."""
    value = strip_wrapping_quotes(raw_value)
    if value == "prod_canary":
        value = "configs/prod_canary.toml"

    profile_path = Path(value)
    candidates: List[Path] = []
    if profile_path.is_absolute():
        candidates.append(profile_path)
    else:
        candidates.append(PROJECT_ROOT / profile_path)
        candidates.append(source_env.resolve().parent / profile_path)
        candidates.append(Path.cwd() / profile_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"unable to resolve canary profile '{raw_value}' from {source_env}"
    )


def rewrite_env_var(path: Path, key: str, value: str) -> None:
    """Replace or append a KEY=VALUE assignment in a shell env file."""
    lines = path.read_text().splitlines()
    new_line = f"{key}={value}"
    replaced = False
    rewritten: List[str] = []

    for line in lines:
        stripped = line.strip()
        candidate = stripped[7:].strip() if stripped.startswith("export ") else stripped
        if candidate.startswith(f"{key}="):
            rewritten.append(new_line)
            replaced = True
        else:
            rewritten.append(line)

    if not replaced:
        rewritten.append(new_line)

    path.write_text("\n".join(rewritten) + "\n")


def materialize_canary_profile(
    env_path: Path,
    config_dir: Path,
    source_env: Path,
    dry_run: bool = False,
) -> Optional[Path]:
    """Snapshot the referenced canary profile and rewrite the env to an absolute path."""
    env_values = load_env_values([env_path])
    raw_profile = env_values.get("PARAPHINA_LIVE_CANARY_PROFILE")
    if not raw_profile:
        return None

    source_profile = resolve_canary_profile_source(raw_profile, source_env)
    source_sha = sha256_file(source_profile)
    snapshot_name = (
        f"{shorten_snapshot_component(env_path.stem, 72)}__"
        f"{shorten_snapshot_component(source_profile.stem, 48)}_"
        f"{source_sha[:12]}{source_profile.suffix}"
    )
    snapshot_dir = config_dir / CANARY_SNAPSHOT_DIR
    snapshot_path = snapshot_dir / snapshot_name

    if dry_run:
        print(f"[dry-run] Would snapshot canary profile: {source_profile} -> {snapshot_path}")
        print(
            f"[dry-run] Would rewrite {env_path.name}: "
            f"PARAPHINA_LIVE_CANARY_PROFILE={snapshot_path}"
        )
        return snapshot_path

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_profile, snapshot_path)
    rewrite_env_var(
        env_path,
        "PARAPHINA_LIVE_CANARY_PROFILE",
        str(snapshot_path),
    )
    print(f"[ok] Snapshotted canary profile: {snapshot_path}")
    print(f"     Source: {source_profile}")
    print(f"     SHA-256: {source_sha}")
    return snapshot_path


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


def sync_live_exec_dropin(config_dir: Path, dry_run: bool = False) -> bool:
    should_enable = live_exec_requested(config_dir)
    changed = False

    if should_enable:
        if dry_run:
            print(f"[dry-run] Would ensure live-exec drop-in: {LIVE_EXEC_DROPIN_PATH}")
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
            print(f"[dry-run] Would remove stale live-exec drop-in: {LIVE_EXEC_DROPIN_PATH}")
        else:
            LIVE_EXEC_DROPIN_PATH.unlink()
            changed = True

    if dry_run or not changed:
        return True

    cmd = ["sudo", "systemctl", "daemon-reload"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(
                f"[error] systemd daemon-reload failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False
    except subprocess.TimeoutExpired:
        print(f"[error] systemd daemon-reload timed out", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"[error] systemctl not found — are you on a systemd host?", file=sys.stderr)
        return False
    return True


def write_stage_overlay(
    config_dir: Path,
    trade_mode: str,
    canary_mode: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """Write a stage overlay file for the next service restart."""
    overlay_path = config_dir / STAGE_OVERLAY_FILE
    lines = [f"PARAPHINA_TRADE_MODE={trade_mode}"]
    lines.append(f"PARAPHINA_CANARY_MODE={1 if canary_mode else 0}")
    if env_overrides:
        for key, value in sorted(env_overrides.items()):
            lines.append(f"{key}={value}")
    overlay_path.write_text("\n".join(lines) + "\n")


def restart_service(
    service: str = SERVICE_NAME,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
    trade_mode_override: Optional[str] = None,
    canary_mode: bool = False,
    env_overrides: Optional[Dict[str, str]] = None,
) -> bool:
    """Restart the systemd service, optionally with stage overrides."""
    overlay_path = config_dir / STAGE_OVERLAY_FILE
    cmd = ["sudo", "systemctl", "restart", service]
    if dry_run:
        if trade_mode_override is not None:
            print(f"[dry-run] Would write stage overlay: {overlay_path}")
            print(f"[dry-run]   PARAPHINA_TRADE_MODE={trade_mode_override}")
            print(f"[dry-run]   PARAPHINA_CANARY_MODE={1 if canary_mode else 0}")
            if env_overrides:
                for key, value in sorted(env_overrides.items()):
                    print(f"[dry-run]   {key}={value}")
        elif overlay_path.exists():
            print(f"[dry-run] Would remove stale overlay: {overlay_path}")
        sync_live_exec_dropin(config_dir=config_dir, dry_run=True)
        print(f"[dry-run] Would execute: {' '.join(cmd)}")
        return True
    try:
        if trade_mode_override is not None:
            write_stage_overlay(
                config_dir=config_dir,
                trade_mode=trade_mode_override,
                canary_mode=canary_mode,
                env_overrides=env_overrides,
            )
        elif overlay_path.exists():
            overlay_path.unlink()
        if not sync_live_exec_dropin(config_dir=config_dir, dry_run=dry_run):
            return False
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[error] Service restart failed: {result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[error] Service restart timed out", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"[error] systemctl not found — are you on a systemd host?", file=sys.stderr)
        return False


# ===========================================================================
# Commands
# ===========================================================================

def cmd_init(args: argparse.Namespace) -> int:
    """Initialise the config directory with a base env file."""
    config_dir = Path(args.config_dir)
    base_env = Path(args.base_env)

    if not base_env.exists():
        print(f"[error] Base env file not found: {base_env}", file=sys.stderr)
        return 1

    if not validate_env_file(base_env):
        print(f"[error] Base env file is not a valid env file: {base_env}", file=sys.stderr)
        return 1

    config_dir.mkdir(parents=True, exist_ok=True)

    # Copy base env as the initial config
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_name = f"baseline_{ts}.env"
    dest = config_dir / dest_name
    shutil.copy2(base_env, dest)

    # Create symlinks
    current_link = config_dir / CURRENT_LINK
    previous_link = config_dir / PREVIOUS_LINK

    if current_link.is_symlink() or current_link.exists():
        current_link.unlink()
    current_link.symlink_to(dest_name)

    if previous_link.is_symlink() or previous_link.exists():
        previous_link.unlink()
    previous_link.symlink_to(dest_name)  # both point to same initially

    # Write initial state
    state = DeployState(
        active_config=dest_name,
        active_config_sha256=sha256_file(dest),
        previous_config=dest_name,
        previous_config_sha256=sha256_file(dest),
        deploy_timestamp=now_iso(),
        current_stage="live",
        git_commit=git_commit_short(),
    )
    save_state(config_dir, state)

    record = DeployRecord(
        timestamp=now_iso(),
        action="init",
        config_file=dest_name,
        config_sha256=sha256_file(dest),
        git_commit=git_commit_short(),
    )
    append_history(config_dir, record)

    print(f"[ok] Initialised {config_dir}")
    print(f"     current.env  -> {dest_name}")
    print(f"     previous.env -> {dest_name}")
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    """Activate a promoted config: rotate symlinks and optionally restart."""
    config_dir = Path(args.config_dir)
    promoted_env = Path(args.env_file)
    stage = getattr(args, "stage", None) or "shadow"
    dry_run = getattr(args, "dry_run", False)

    if not config_dir.exists():
        print(f"[error] Config dir does not exist: {config_dir}", file=sys.stderr)
        print("        Run 'config_manager.py init' first.", file=sys.stderr)
        return 1

    if not promoted_env.exists():
        print(f"[error] Promoted env file not found: {promoted_env}", file=sys.stderr)
        return 1

    if not validate_env_file(promoted_env):
        print(f"[error] Promoted env file is not valid: {promoted_env}", file=sys.stderr)
        return 1

    state = load_state(config_dir)

    # Copy the promoted env into the config dir with a timestamp
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_name = f"promoted_{ts}_{promoted_env.stem}.env"
    dest = config_dir / dest_name
    shutil.copy2(promoted_env, dest)
    if not sync_runtime_configs(dry_run=dry_run):
        return 1
    try:
        materialize_canary_profile(dest, config_dir, promoted_env, dry_run=dry_run)
    except FileNotFoundError as err:
        print(f"[error] {err}", file=sys.stderr)
        return 1
    new_sha = sha256_file(dest)

    # Rotate symlinks: previous -> old current, current -> new
    current_link = config_dir / CURRENT_LINK
    previous_link = config_dir / PREVIOUS_LINK

    old_current = resolve_symlink_target(config_dir, CURRENT_LINK)

    if previous_link.is_symlink() or previous_link.exists():
        previous_link.unlink()
    if old_current:
        previous_link.symlink_to(old_current)
    else:
        previous_link.symlink_to(dest_name)

    if current_link.is_symlink() or current_link.exists():
        current_link.unlink()
    current_link.symlink_to(dest_name)

    # Update state
    state.previous_config = old_current or state.active_config
    state.previous_config_sha256 = state.active_config_sha256
    state.active_config = dest_name
    state.active_config_sha256 = new_sha
    state.deploy_timestamp = now_iso()
    state.current_stage = stage
    state.git_commit = git_commit_short()
    save_state(config_dir, state)

    record = DeployRecord(
        timestamp=now_iso(),
        action="activate",
        config_file=dest_name,
        config_sha256=new_sha,
        previous_config=old_current,
        git_commit=git_commit_short(),
        stage=stage,
    )
    append_history(config_dir, record)

    print(f"[ok] Activated config: {dest_name}")
    print(f"     current.env  -> {dest_name}")
    print(f"     previous.env -> {old_current or dest_name}")
    print(f"     SHA-256: {new_sha}")
    print(f"     Stage: {stage}")

    if getattr(args, "restart", False):
        print(f"[info] Restarting {SERVICE_NAME}...")
        stage_trade_mode = "live" if stage in {"canary", "live"} else stage
        stage_canary_mode = stage == "canary"
        stage_env_overrides = None
        if stage in {"canary", "live"}:
            stage_env_overrides = {
                "PARAPHINA_LIVE_EXEC_ENABLE": "1",
                "PARAPHINA_LIVE_EXECUTION_CONFIRM": "YES",
            }
        if not restart_service(
            config_dir=config_dir,
            dry_run=dry_run,
            trade_mode_override=stage_trade_mode,
            canary_mode=stage_canary_mode,
            env_overrides=stage_env_overrides,
        ):
            return 1

    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Rollback to the previous known-good config."""
    config_dir = Path(args.config_dir)
    reason = getattr(args, "reason", None) or "manual rollback"
    dry_run = getattr(args, "dry_run", False)

    if not config_dir.exists():
        print(f"[error] Config dir does not exist: {config_dir}", file=sys.stderr)
        return 1

    state = load_state(config_dir)

    if not state.previous_config:
        print("[error] No previous config to rollback to.", file=sys.stderr)
        return 1

    # Guard against double rollback (active == previous).
    if state.active_config == state.previous_config:
        print("[error] Active config is already the previous config — nothing to roll back to.", file=sys.stderr)
        print(f"        Active:   {state.active_config}", file=sys.stderr)
        print(f"        Previous: {state.previous_config}", file=sys.stderr)
        return 1

    prev_file = config_dir / state.previous_config
    if not prev_file.exists():
        print(f"[error] Previous config file missing: {prev_file}", file=sys.stderr)
        return 1

    if not sync_runtime_configs(dry_run=dry_run):
        return 1

    # Verify integrity
    actual_sha = sha256_file(prev_file)
    if state.previous_config_sha256 and actual_sha != state.previous_config_sha256:
        print(f"[error] Previous config SHA-256 mismatch!", file=sys.stderr)
        print(f"        Expected: {state.previous_config_sha256}", file=sys.stderr)
        print(f"        Actual:   {actual_sha}", file=sys.stderr)
        return 1

    current_link = config_dir / CURRENT_LINK
    previous_link = config_dir / PREVIOUS_LINK
    old_current = resolve_symlink_target(config_dir, CURRENT_LINK)

    # Swap current symlink to previous config
    if current_link.is_symlink() or current_link.exists():
        current_link.unlink()
    current_link.symlink_to(state.previous_config)

    # Update previous.env to point to the config we're rolling back from,
    # so a subsequent rollback can undo this one.
    if previous_link.is_symlink() or previous_link.exists():
        previous_link.unlink()
    if old_current:
        previous_link.symlink_to(old_current)

    # Update state
    rolled_back_from = state.active_config
    rolled_back_from_sha = state.active_config_sha256
    state.active_config = state.previous_config
    state.active_config_sha256 = state.previous_config_sha256
    state.previous_config = rolled_back_from
    state.previous_config_sha256 = rolled_back_from_sha
    rollback_stage = "live" if live_exec_requested(config_dir) else "shadow"
    state.current_stage = rollback_stage
    state.rollback_count += 1
    state.last_rollback_timestamp = now_iso()
    state.last_rollback_reason = reason
    state.deploy_timestamp = now_iso()
    save_state(config_dir, state)

    record = DeployRecord(
        timestamp=now_iso(),
        action="rollback",
        config_file=state.active_config or "unknown",
        config_sha256=actual_sha,
        previous_config=rolled_back_from,
        reason=reason,
        git_commit=git_commit_short(),
    )
    append_history(config_dir, record)

    print(f"[ok] Rolled back to: {state.active_config}")
    print(f"     Rolled back from: {rolled_back_from}")
    print(f"     Reason: {reason}")
    if rollback_stage != "live":
        print(
            "     Restart stage: shadow "
            "(restored env is not live-exec-enabled)"
        )

    if not getattr(args, "no_restart", False):
        print(f"[info] Restarting {SERVICE_NAME}...")
        restart_trade_mode = None if rollback_stage == "live" else rollback_stage
        if not restart_service(
            config_dir=config_dir,
            dry_run=dry_run,
            trade_mode_override=restart_trade_mode,
        ):
            return 1

    return 0


def cmd_update_stage(args: argparse.Namespace) -> int:
    """Update the current deploy stage in the state file."""
    config_dir = Path(args.config_dir)
    stage = args.stage

    if stage not in ("shadow", "paper", "canary", "live"):
        print(f"[error] Invalid stage: {stage}. Must be shadow|paper|canary|live", file=sys.stderr)
        return 1

    state = load_state(config_dir)
    state.current_stage = stage
    save_state(config_dir, state)
    print(f"[ok] Stage updated to: {stage}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show current deploy state."""
    config_dir = Path(args.config_dir)

    if not config_dir.exists():
        print(f"Config dir does not exist: {config_dir}")
        return 1

    state = load_state(config_dir)

    # Also resolve actual symlinks for verification
    actual_current = resolve_symlink_target(config_dir, CURRENT_LINK)
    actual_previous = resolve_symlink_target(config_dir, PREVIOUS_LINK)

    print(f"Config directory: {config_dir}")
    print(f"")
    print(f"Active config:    {state.active_config or '(none)'}")
    print(f"  SHA-256:        {state.active_config_sha256 or '(none)'}")
    print(f"  Symlink:        current.env -> {actual_current or '(broken)'}")
    print(f"")
    print(f"Previous config:  {state.previous_config or '(none)'}")
    print(f"  SHA-256:        {state.previous_config_sha256 or '(none)'}")
    print(f"  Symlink:        previous.env -> {actual_previous or '(broken)'}")
    print(f"")
    print(f"Deploy timestamp: {state.deploy_timestamp or '(never)'}")
    print(f"Current stage:    {state.current_stage or '(unknown)'}")
    print(f"Git commit:       {state.git_commit or '(unknown)'}")
    print(f"Rollback count:   {state.rollback_count}")
    if state.last_rollback_timestamp:
        print(f"Last rollback:    {state.last_rollback_timestamp}")
        print(f"  Reason:         {state.last_rollback_reason}")

    # Verify consistency
    issues = []
    if actual_current != state.active_config:
        issues.append(f"current.env symlink ({actual_current}) != state ({state.active_config})")
    if actual_current and (config_dir / actual_current).exists():
        actual_sha = sha256_file(config_dir / actual_current)
        if state.active_config_sha256 and actual_sha != state.active_config_sha256:
            issues.append(f"Active config SHA-256 mismatch (file changed on disk?)")

    if issues:
        print(f"\n[warn] Consistency issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print(f"\n[ok] State consistent.")
    return 0


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="config_manager",
        description="Paraphina config lifecycle manager",
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help=f"Config directory (default: {DEFAULT_CONFIG_DIR})",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialise config directory")
    p_init.add_argument("--base-env", required=True, help="Path to baseline .env file")

    # activate
    p_act = sub.add_parser("activate", help="Activate a promoted config")
    p_act.add_argument("env_file", help="Path to the promoted .env file")
    p_act.add_argument("--stage", default="shadow", help="Initial deploy stage")
    p_act.add_argument("--restart", action="store_true", help="Restart service after activation")
    p_act.add_argument("--dry-run", action="store_true", help="Print actions without executing")

    # rollback
    p_rb = sub.add_parser("rollback", help="Rollback to previous config")
    p_rb.add_argument("--reason", default="manual rollback", help="Reason for rollback")
    p_rb.add_argument("--no-restart", action="store_true", help="Skip service restart")
    p_rb.add_argument("--dry-run", action="store_true", help="Print actions without executing")

    # update-stage
    p_stage = sub.add_parser("update-stage", help="Update deploy stage")
    p_stage.add_argument("stage", choices=["shadow", "paper", "canary", "live"])

    # status
    sub.add_parser("status", help="Show current deploy state")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "init": cmd_init,
        "activate": cmd_activate,
        "rollback": cmd_rollback,
        "update-stage": cmd_update_stage,
        "status": cmd_status,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
