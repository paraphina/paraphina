#!/usr/bin/env python3
"""Terminal dashboard for paraphina_live telemetry.

Displays a colour-coded, Unicode-styled live view of venue status,
positions, fills, cancels, and kill events.  All rendering is pure
display logic – no market-making behaviour is changed.
"""
from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import re
import signal
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Mapping

try:
    from rich import box
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

try:
    import fcntl
    import termios
    import tty
except ImportError:  # pragma: no cover - non-Unix fallback
    fcntl = None
    termios = None
    tty = None

# ── ANSI styling ──────────────────────────────────────────────────────────────

_NO_COLOR = False  # flipped by --no-color / NO_COLOR env / non-TTY

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_VENUE_CODE_MAP = {
    "hyperliquid": "HL",
    "aster": "AS",
    "extended": "EX",
    "paradex": "PA",
    "lighter": "LG",
}

_TAB_TAPE = 0
_TAB_KEYS = 1
_TAB_ALERTS = 2

_PAGE_SIMPLE = "simple"
_PAGE_EXPANDED = "expanded"

_WATCH_REPO_ROOT = Path(__file__).resolve().parent.parent
_SYSTEMD_TELEMETRY_PATH = Path("/var/lib/paraphina/out/telemetry.jsonl")
_CURRENT_RUN_POINTER_PATH = Path("/tmp/paraphina_current_run.json")
_CURRENT_RUNS_DIR = Path("/tmp/paraphina_current_runs")
_SHADOW_LATEST_PATH = Path("/tmp/paraphina_shadow_latest")
_SHADOW_LAST_OUTDIR_PATH = Path("/tmp/paraphina_last_outdir.txt")
_SHADOW_STATE_PATH = Path("/tmp/paraphina_shadow_runner.state")
_SHADOW_PID_PATH = Path("/tmp/paraphina_shadow_runner.pid")
_SOURCE_OWNER_PHASE51_ROOT = Path("/home/ubuntu/source_owner_inbox/phase51")
_AUTO_TARGET_GLOB_ROOTS: tuple[tuple[Path, tuple[str, ...]], ...] = (
    (Path("/tmp"), ("*/telemetry.jsonl", "*/*/telemetry.jsonl")),
    (_WATCH_REPO_ROOT / "runs", ("*/telemetry.jsonl", "*/*/telemetry.jsonl")),
    (_SOURCE_OWNER_PHASE51_ROOT, ("*/telemetry.jsonl", "*/*/telemetry.jsonl")),
)


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
        description="Terminal dashboard for paraphina_live telemetry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Freshness semantics:\n"
            "  ageE = event age (venue timestamp -> rx)\n"
            "  ageA = apply age (rx -> publish/apply)\n"
            "Flash semantics:\n"
            "  BUY flashes green, SELL flashes red via TTL highlight (no ANSI blink).\n"
            "Path helpers:\n"
            "  --current-run   -> follows the active live/shadow run\n"
            "  --current-shadow -> follows the active repo-managed shadow run\n"
            "  --run-dir DIR   -> reads DIR/telemetry.jsonl\n"
            "  --latest        -> follows the freshest automatic target\n"
        ),
    )
    path_group = parser.add_mutually_exclusive_group(required=True)
    path_group.add_argument("--telemetry", help="Path to telemetry.jsonl")
    path_group.add_argument(
        "--run-dir",
        help="Run directory containing telemetry.jsonl",
    )
    path_group.add_argument(
        "--current-run",
        action="store_true",
        help="Follow the active live/shadow run",
    )
    path_group.add_argument(
        "--current-shadow",
        action="store_true",
        help="Follow the active repo-managed shadow run",
    )
    path_group.add_argument(
        "--latest",
        action="store_true",
        help="Follow the freshest automatic telemetry target",
    )
    parser.add_argument("--refresh-ms", type=int, default=250)
    parser.add_argument(
        "--render-ms",
        type=int,
        default=None,
        help=(
            "Render cadence in milliseconds (default: refresh-ms, or "
            "max(refresh-ms,120) when VS Code mode auto-detects)"
        ),
    )
    parser.add_argument("--max-events", type=int, default=50)
    parser.add_argument(
        "--sort",
        choices=("agee", "stale", "venue"),
        default="agee",
        help="Venue sort mode for rich UI (default: agee/worst-first)",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="Use the legacy classic renderer",
    )
    parser.add_argument(
        "--page",
        choices=(_PAGE_SIMPLE, _PAGE_EXPANDED),
        default=_PAGE_SIMPLE,
        help="Startup page for rich UI (default: simple)",
    )
    parser.add_argument(
        "--flash-ms",
        type=int,
        default=650,
        help="BUY/SELL flash TTL in milliseconds (default: 650)",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable coloured output"
    )
    parser.add_argument(
        "--health-url", default="http://127.0.0.1:9898",
        help="Base URL for /health/detail endpoint (default: http://127.0.0.1:9898)",
    )
    parser.add_argument(
        "--config-dir", default="/etc/paraphina",
        help="Config directory for deploy state (default: /etc/paraphina)",
    )
    parser.add_argument(
        "--no-deploy-state", action="store_true",
        help="Disable deploy state panel",
    )
    parser.add_argument(
        "--layout-debug",
        action="store_true",
        help="Print rich layout row allocation once",
    )
    vscode_group = parser.add_mutually_exclusive_group()
    vscode_group.add_argument(
        "--vscode",
        action="store_true",
        help="Force VS Code render mode on",
    )
    vscode_group.add_argument(
        "--no-vscode",
        action="store_true",
        help="Force VS Code render mode off",
    )
    return parser.parse_args()


def _auto_target_candidates() -> list[Path]:
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    latest = _SHADOW_LATEST_PATH
    if latest.exists():
        if latest.is_dir():
            add_candidate(latest / "telemetry.jsonl")
        else:
            add_candidate(latest)

    last_outdir = _SHADOW_LAST_OUTDIR_PATH
    if last_outdir.exists():
        try:
            outdir = last_outdir.read_text(encoding="utf-8").strip()
        except OSError:
            outdir = ""
        if outdir:
            add_candidate(Path(outdir) / "telemetry.jsonl")

    add_candidate(_SYSTEMD_TELEMETRY_PATH)
    add_candidate(Path("/tmp/paraphina_live_shadow/telemetry.jsonl"))

    for root, patterns in _AUTO_TARGET_GLOB_ROOTS:
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                matches = root.glob(pattern)
            except OSError:
                continue
            for candidate in matches:
                if candidate.is_file():
                    add_candidate(candidate)
    return candidates


def _shadow_telemetry_candidates() -> list[Path]:
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        normalized = _normalize_path(path)
        if normalized not in candidates:
            candidates.append(normalized)

    latest = _SHADOW_LATEST_PATH
    if latest.exists():
        if latest.is_dir():
            add_candidate(latest / "telemetry.jsonl")
        else:
            add_candidate(latest)

    last_outdir = _SHADOW_LAST_OUTDIR_PATH
    if last_outdir.exists():
        try:
            outdir = last_outdir.read_text(encoding="utf-8").strip()
        except OSError:
            outdir = ""
        if outdir:
            add_candidate(Path(outdir) / "telemetry.jsonl")
    return candidates


def _resolve_shadow_state_target(state_path: Path) -> Path:
    data = _read_kv_file(state_path)
    state = data.get("state") or "unknown"
    runner_pid = _safe_parse_int(data.get("runner_pid"))
    telemetry_raw = data.get("telemetry_path")
    outdir = data.get("outdir")
    if telemetry_raw:
        telemetry_path = _normalize_path(telemetry_raw)
    elif outdir:
        telemetry_path = _normalize_path(Path(outdir) / "telemetry.jsonl")
    else:
        raise FileNotFoundError(
            f"shadow runner state missing telemetry target: {state_path}"
        )
    if state not in {"running", "starting"}:
        raise FileNotFoundError(
            f"shadow runner is not active (state={state}) per {state_path}"
        )
    if not _pid_is_paraphina_runner(runner_pid):
        runner_text = "unknown" if runner_pid is None else str(runner_pid)
        raise FileNotFoundError(
            f"shadow runner pid {runner_text} is not active per {state_path}"
        )
    return telemetry_path


@dataclass
class CurrentRunRecord:
    pid: int
    telemetry_path: Path
    started_at_unix_ms: int = 0
    trade_mode: str | None = None
    manifest_path: str | None = None


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_current_run_record(
    raw: Mapping[str, Any] | None, source_path: Path
) -> CurrentRunRecord | None:
    if not isinstance(raw, Mapping):
        return None
    pid = safe_int(raw.get("pid"))
    telemetry_raw = raw.get("telemetry_path")
    if pid is None or not isinstance(telemetry_raw, str) or not telemetry_raw.strip():
        return None
    if not _pid_is_paraphina_runner(pid):
        return None
    started_at_unix_ms = safe_int(raw.get("started_at_unix_ms")) or 0
    trade_mode = raw.get("trade_mode")
    manifest_path = raw.get("manifest_path")
    return CurrentRunRecord(
        pid=pid,
        telemetry_path=_normalize_path(telemetry_raw),
        started_at_unix_ms=started_at_unix_ms,
        trade_mode=str(trade_mode) if trade_mode is not None else None,
        manifest_path=str(manifest_path) if manifest_path is not None else str(source_path),
    )


def _resolve_current_run_from_registry() -> CurrentRunRecord | None:
    pointer_record = _parse_current_run_record(
        _read_json_file(_CURRENT_RUN_POINTER_PATH),
        _CURRENT_RUN_POINTER_PATH,
    )
    if pointer_record is not None:
        return pointer_record

    candidates: list[CurrentRunRecord] = []
    try:
        entries = sorted(_CURRENT_RUNS_DIR.glob("*.json"))
    except OSError:
        entries = []
    for candidate in entries:
        record = _parse_current_run_record(_read_json_file(candidate), candidate)
        if record is not None:
            candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda record: (record.started_at_unix_ms, record.pid))


def resolve_current_run_telemetry_path() -> Path:
    current_record = _resolve_current_run_from_registry()
    if current_record is not None:
        return current_record.telemetry_path

    current_path = _infer_current_telemetry_path_from_processes()
    if current_path is not None:
        return current_path

    raise FileNotFoundError(
        "no active current run found; use --latest for the freshest saved artifact "
        "or --run-dir/--telemetry for an explicit target"
    )


def resolve_current_shadow_telemetry_path() -> Path:
    if _SHADOW_STATE_PATH.exists():
        return _resolve_shadow_state_target(_SHADOW_STATE_PATH)

    for telemetry_path in _shadow_telemetry_candidates():
        status = load_runner_status(telemetry_path)
        if status is not None and status.alive:
            return telemetry_path

    raise FileNotFoundError(
        "no active shadow run found; checked "
        f"{_SHADOW_STATE_PATH}, {_SHADOW_LAST_OUTDIR_PATH}, and {_SHADOW_LATEST_PATH}"
    )


def resolve_latest_telemetry_path() -> Path:
    current_path = _infer_current_telemetry_path_from_processes()
    if current_path is not None:
        return current_path

    best_path: Path | None = None
    best_mtime_ns = -1
    for candidate in _auto_target_candidates():
        try:
            stat_result = candidate.stat()
        except OSError:
            continue
        if stat_result.st_mtime_ns > best_mtime_ns:
            best_path = candidate
            best_mtime_ns = stat_result.st_mtime_ns
    if best_path is not None:
        return best_path
    raise FileNotFoundError(
        "no telemetry source found; checked current-run markers, "
        "/var/lib/paraphina/out/telemetry.jsonl, /tmp telemetry runs, "
        "repo runs/*/telemetry.jsonl, and source_owner_inbox/phase51 telemetry"
    )


def resolve_telemetry_path(args: argparse.Namespace) -> Path:
    if args.telemetry:
        return Path(args.telemetry)
    if args.run_dir:
        return Path(args.run_dir) / "telemetry.jsonl"
    if args.current_run:
        return resolve_current_run_telemetry_path()
    if args.current_shadow:
        return resolve_current_shadow_telemetry_path()
    if args.latest:
        return resolve_latest_telemetry_path()
    raise ValueError("no telemetry path source selected")


def detect_vscode_terminal(env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    if source.get("TERM_PROGRAM") == "vscode":
        return True
    return any(key.startswith("VSCODE_") for key in source)


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


def wall_clock_utc() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def heartbeat_glyph(frame_count: int) -> str:
    glyphs = "|/-\\"
    return glyphs[frame_count % len(glyphs)]


def source_age_label(state: "WatchState", now_mono: float) -> tuple[str, str]:
    if state._last_update_mono is None:
        return ("src n/a", "dim")
    age_s = max(0.0, now_mono - state._last_update_mono)
    if age_s < 1.0:
        return (f"src {int(age_s * 1000)}ms", "green")
    if age_s < 5.0:
        return (f"src {age_s:.1f}s", "white")
    if age_s < 15.0:
        return (f"src {age_s:.1f}s", "yellow")
    return (f"src {age_s:.1f}s", "bold red")


@dataclass
class RunnerStatus:
    state: str = "unknown"
    runner_pid: int | None = None
    supervisor_pid: int | None = None
    session_name: str | None = None
    outdir: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None
    exit_code: int | None = None
    signal_num: int | None = None
    alive: bool = False
    status_path: str | None = None


def _safe_parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_kv_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return data
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _read_cmdline_tokens(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    parts = [part.decode("utf-8", errors="ignore") for part in raw.split(b"\x00") if part]
    return parts or None


def _pid_is_paraphina_runner(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = _read_cmdline_tokens(pid)
    if not cmdline:
        return False
    return any("paraphina_live" in token for token in cmdline)


def _extract_cmd_arg(cmdline: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for idx, token in enumerate(cmdline):
        if token == flag and idx + 1 < len(cmdline):
            return cmdline[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _iter_running_paraphina_live() -> Iterable[tuple[int, list[str]]]:
    proc_root = Path("/proc")
    try:
        proc_entries = list(proc_root.iterdir())
    except OSError:
        return
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmdline = _read_cmdline_tokens(pid)
        if not cmdline:
            continue
        if any("paraphina_live" in token for token in cmdline):
            yield pid, cmdline


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _infer_current_telemetry_path_from_processes() -> Path | None:
    telemetry_paths: list[Path] = []

    def add_path(path: Path) -> None:
        normalized = _normalize_path(path)
        if normalized not in telemetry_paths:
            telemetry_paths.append(normalized)

    for _pid, cmdline in _iter_running_paraphina_live():
        outdir = _extract_cmd_arg(cmdline, "--out-dir")
        if outdir:
            add_path(Path(outdir) / "telemetry.jsonl")
        else:
            add_path(_SYSTEMD_TELEMETRY_PATH)

    if len(telemetry_paths) == 1:
        return telemetry_paths[0]
    return None


def _infer_runner_status_from_processes(telemetry_path: Path) -> RunnerStatus | None:
    target_outdir = _normalize_path(telemetry_path.parent)
    default_outdir = _normalize_path(_SYSTEMD_TELEMETRY_PATH.parent)
    fallback_pid: int | None = None
    for pid, cmdline in _iter_running_paraphina_live():
        outdir = _extract_cmd_arg(cmdline, "--out-dir")
        if outdir is not None:
            if _normalize_path(outdir) != target_outdir:
                continue
            return RunnerStatus(
                state="running",
                runner_pid=pid,
                outdir=str(target_outdir),
                alive=True,
                status_path=f"/proc/{pid}/cmdline",
            )
        if target_outdir == default_outdir and fallback_pid is None:
            fallback_pid = pid
    if fallback_pid is None:
        return None
    return RunnerStatus(
        state="running",
        runner_pid=fallback_pid,
        outdir=str(target_outdir),
        alive=True,
        status_path=f"/proc/{fallback_pid}/cmdline",
    )


def load_runner_status(telemetry_path: Path) -> RunnerStatus | None:
    candidates = [
        telemetry_path.parent / "runner.state",
        _SHADOW_STATE_PATH,
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = _read_kv_file(candidate)
        status = RunnerStatus(
            state=data.get("state") or "unknown",
            runner_pid=_safe_parse_int(data.get("runner_pid")),
            supervisor_pid=_safe_parse_int(data.get("supervisor_pid")),
            session_name=data.get("session_name") or None,
            outdir=data.get("outdir") or None,
            started_at=data.get("started_at") or None,
            stopped_at=data.get("stopped_at") or None,
            exit_code=_safe_parse_int(data.get("exit_code")),
            signal_num=_safe_parse_int(data.get("signal")),
            status_path=str(candidate),
        )
        if status.outdir and status.outdir != str(telemetry_path.parent):
            continue
        status.alive = _pid_is_paraphina_runner(status.runner_pid)
        return status

    local_pid_file = telemetry_path.parent / "runner.pid"
    if local_pid_file.exists():
        pid = _safe_parse_int(local_pid_file.read_text(encoding="utf-8").strip())
        return RunnerStatus(
            state="running" if _pid_is_paraphina_runner(pid) else "unknown",
            runner_pid=pid,
            outdir=str(telemetry_path.parent),
            alive=_pid_is_paraphina_runner(pid),
            status_path=str(local_pid_file),
        )

    pid_file = _SHADOW_PID_PATH
    if pid_file.exists():
        try:
            last_outdir = _SHADOW_LAST_OUTDIR_PATH.read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            last_outdir = ""
        if last_outdir == str(telemetry_path.parent):
            pid = _safe_parse_int(pid_file.read_text(encoding="utf-8").strip())
            return RunnerStatus(
                state="running" if _pid_is_paraphina_runner(pid) else "unknown",
                runner_pid=pid,
                outdir=str(telemetry_path.parent),
                alive=_pid_is_paraphina_runner(pid),
                status_path=str(pid_file),
            )
    return _infer_runner_status_from_processes(telemetry_path)


def runner_status_label(state: "WatchState") -> tuple[str, str]:
    status = state.runner_status
    if status is None:
        return ("runner ?", "dim")
    if status.state == "no_active":
        return ("no active run", "bold red")
    pid_text = f" pid {status.runner_pid}" if status.runner_pid is not None else ""
    if status.alive:
        if status.state == "starting":
            return (f"runner starting{pid_text}", "yellow")
        if status.state == "stopping":
            return (f"runner stopping{pid_text}", "yellow")
        return (f"runner live{pid_text}", "green")
    if status.state == "exited":
        if status.signal_num is not None:
            return (f"runner dead sig {status.signal_num}", "bold red")
        if status.exit_code is not None:
            return (f"runner dead exit {status.exit_code}", "bold red")
    if status.state == "stopping":
        return (f"runner stopping{pid_text}", "yellow")
    if status.runner_pid is not None:
        return (f"runner dead{pid_text}", "bold red")
    return ("runner dead", "bold red")


def seed_state_source_age_from_path(state: "WatchState", telemetry_path: Path) -> None:
    if state.last_record is None:
        return
    try:
        age_s = max(0.0, time.time() - telemetry_path.stat().st_mtime)
    except OSError:
        return
    state._last_update_mono = max(0.0, time.monotonic() - age_s)


def _safe_float_from_any(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_balance_pnl_state(run_dir: Path) -> BalancePnlState:
    comparison_path = run_dir / "balance_snapshot_comparison.json"
    if comparison_path.exists():
        try:
            payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return BalancePnlState(status="unreadable", source_path=str(comparison_path))
        total = payload.get("total") if isinstance(payload, dict) else None
        total = total if isinstance(total, dict) else {}
        return BalancePnlState(
            status="available",
            delta_usd=_safe_float_from_any(total.get("delta_usd")),
            pre_usd=_safe_float_from_any(total.get("pre_usd")),
            post_usd=_safe_float_from_any(total.get("post_usd")),
            venue_count=safe_int(payload.get("venue_count")) if isinstance(payload, dict) else None,
            generated_at_utc=(
                str(payload.get("generated_at_utc"))
                if isinstance(payload, dict) and payload.get("generated_at_utc") is not None
                else None
            ),
            source_path=str(comparison_path),
        )

    pre_snapshot_path = run_dir / "balance_pre_snapshot.json"
    if pre_snapshot_path.exists():
        try:
            payload = json.loads(pre_snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return BalancePnlState(status="pending_post", source_path=str(pre_snapshot_path))
        return BalancePnlState(
            status="pending_post",
            pre_usd=_safe_float_from_any(payload.get("total_balance_usd")) if isinstance(payload, dict) else None,
            venue_count=safe_int(payload.get("venue_count")) if isinstance(payload, dict) else None,
            generated_at_utc=(
                str(payload.get("captured_at_utc"))
                if isinstance(payload, dict) and payload.get("captured_at_utc") is not None
                else None
            ),
            source_path=str(pre_snapshot_path),
        )
    return BalancePnlState(status="missing")


def refresh_balance_pnl_from_run_dir(
    state: "WatchState",
    run_dir: Path,
    *,
    now_mono: float | None = None,
    force: bool = False,
) -> None:
    now = time.monotonic() if now_mono is None else now_mono
    run_dir_text = str(run_dir)
    if (
        not force
        and state.balance_pnl_run_dir == run_dir_text
        and now - state._balance_pnl_last_refresh_mono < 1.0
    ):
        return
    state.balance_pnl = _load_balance_pnl_state(run_dir)
    state.balance_pnl_run_dir = run_dir_text
    state._balance_pnl_last_refresh_mono = now


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


def venue_code(venue_id: str) -> str:
    return _VENUE_CODE_MAP.get(venue_id, (venue_id[:2] if venue_id else "??").upper())


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def compute_delta_mid_bps_map(
    venue_ids: list[str],
    venue_status: list[Any],
    venue_mid: list[Any],
) -> tuple[float | None, dict[str, float | None]]:
    healthy_mids: list[float] = []
    parsed_mid: dict[str, float | None] = {}
    parsed_status: dict[str, str] = {}
    for idx, venue_id in enumerate(venue_ids):
        status = (
            str(venue_status[idx])
            if isinstance(venue_status, list) and idx < len(venue_status)
            else "Unknown"
        )
        mid = (
            safe_float(venue_mid[idx])
            if isinstance(venue_mid, list) and idx < len(venue_mid)
            else None
        )
        parsed_mid[venue_id] = mid
        parsed_status[venue_id] = status
        if status == "Healthy" and mid is not None:
            healthy_mids.append(mid)

    median_mid = _median(healthy_mids)
    out: dict[str, float | None] = {}
    for venue_id in venue_ids:
        mid = parsed_mid.get(venue_id)
        status = parsed_status.get(venue_id, "Unknown")
        if status != "Healthy" or mid is None or median_mid is None or median_mid == 0:
            out[venue_id] = None
            continue
        out[venue_id] = ((mid - median_mid) / median_mid) * 10000.0
    return median_mid, out


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


# ── Deploy state & health detail ──────────────────────────────────────────────


def read_deploy_state(config_dir: str) -> dict[str, Any] | None:
    """Read deploy_state.json from the config directory."""
    state_path = Path(config_dir) / "deploy_state.json"
    try:
        if state_path.exists():
            return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return None


def fetch_health_detail(base_url: str) -> dict[str, Any] | None:
    """Fetch /health/detail JSON (best-effort, no crash on failure)."""
    import urllib.request
    import urllib.error

    url = f"{base_url}/health/detail"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        pass
    return None


def color_stage(stage: str | None) -> str:
    """Colour-code the deploy stage."""
    if not stage:
        return styled("unknown", S.GRAY)
    stage_colors = {
        "shadow": S.BLUE,
        "paper": S.YELLOW,
        "canary": S.MAGENTA,
        "live": S.GREEN,
    }
    color = stage_colors.get(stage, S.GRAY)
    return styled(stage, color, S.BOLD)


def render_deploy_section(
    config_dir: str,
    health_url: str,
) -> list[str]:
    """Render the deploy state section for the dashboard."""
    lines: list[str] = []
    deploy = read_deploy_state(config_dir)
    health = fetch_health_detail(health_url)

    if deploy is None and health is None:
        return []

    lines.append("")
    lines.append(_section("Deploy"))

    if deploy:
        active = deploy.get("active_config", "n/a")
        stage = deploy.get("current_stage")
        rollbacks = deploy.get("rollback_count", 0)
        ts = deploy.get("deploy_timestamp", "")
        # Shorten timestamp for display
        if ts and "T" in ts:
            ts = ts.split("T")[1][:8] + "Z"

        rb_str = styled(str(rollbacks), S.B_RED if rollbacks > 0 else S.GREEN)
        deploy_line = (
            f"  {_LABEL('config')} {styled(active, S.B_WHITE)}   "
            f"{_LABEL('stage')} {color_stage(stage)}   "
            f"{_LABEL('rollbacks')} {rb_str}"
        )
        if ts:
            deploy_line += f"   {_LABEL('deployed')} {styled(ts, S.DIM)}"
        lines.append(deploy_line)

        last_rb_reason = deploy.get("last_rollback_reason")
        if rollbacks > 0 and last_rb_reason:
            lines.append(
                f"  {_LABEL('last_rollback')} {styled(last_rb_reason, S.B_RED)}"
            )
    else:
        lines.append(f"  {styled('(no deploy state)', S.GRAY)}")

    if health:
        uptime = health.get("uptime_seconds", 0)
        ticks = health.get("tick_count", 0)
        errors = health.get("error_count", 0)
        recon = health.get("reconcile_mismatch_count", 0)
        kills = health.get("kill_events_present", False)
        config_id = health.get("config_id", "")

        err_str = styled(str(errors), S.B_RED if errors > 5 else S.GREEN)
        recon_str = styled(str(recon), S.B_RED if recon > 0 else S.GREEN)
        kill_str = styled("YES", S.B_RED, S.BOLD) if kills else styled("no", S.GREEN)

        health_line = (
            f"  {_LABEL('uptime')} {uptime}s   "
            f"{_LABEL('ticks')} {ticks}   "
            f"{_LABEL('errors')} {err_str}   "
            f"{_LABEL('recon')} {recon_str}   "
            f"{_LABEL('kills')} {kill_str}"
        )
        lines.append(health_line)

        # Config mismatch detection
        if deploy and config_id:
            active_config = deploy.get("active_config", "")
            if active_config and config_id and config_id not in active_config:
                lines.append(
                    f"  {styled('CONFIG MISMATCH', S.B_RED, S.BOLD)} "
                    f"state={active_config} process={config_id}"
                )
    else:
        lines.append(f"  {_LABEL('health')} {styled('offline', S.B_RED)}")

    return lines


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


# ── Telemetry parsing ─────────────────────────────────────────────────────────


def parse_lines(path: Path, max_events: int) -> list[dict[str, Any]]:
    records: Deque[dict[str, Any]] = deque(maxlen=max_events)
    if not path.exists():
        return []
    try:
        # Read only the tail so startup remains fast on large telemetry files.
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size <= 0:
                return []
            data = b""
            pos = size
            block_size = 64 * 1024
            target = max(4, max_events * 2)
            lines: list[bytes] = []
            while pos > 0 and len(lines) <= target:
                step = block_size if pos >= block_size else pos
                pos -= step
                handle.seek(pos, os.SEEK_SET)
                data = handle.read(step) + data
                lines = data.splitlines()
            tail_lines = lines[-max_events:]
        for raw_line in tail_lines:
            line = raw_line.decode("utf-8", errors="replace").strip()
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


# ── State tracking ────────────────────────────────────────────────────────────


@dataclass
class EventLog:
    fills: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    cancels: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    kills: Deque[str] = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class TapeEvent:
    event_id: str
    kind: str
    text: str
    venue_id: str | None = None
    ts_ms: int | None = None
    flash_until_mono: float = 0.0


@dataclass
class BalancePnlState:
    status: str = "missing"
    delta_usd: float | None = None
    pre_usd: float | None = None
    post_usd: float | None = None
    venue_count: int | None = None
    generated_at_utc: str | None = None
    source_path: str | None = None


@dataclass
class WatchState:
    sort_mode: str = "agee"
    flash_ttl_s: float = 0.65
    history_len: int = 90
    last_record: dict[str, Any] | None = None
    venue_ids: list[str] = field(default_factory=list)
    last_fill_ms: dict[str, int] = field(default_factory=dict)
    events: EventLog = field(default_factory=EventLog)
    tape: Deque[TapeEvent] = field(default_factory=lambda: deque(maxlen=50))
    alerts: Deque[str] = field(default_factory=lambda: deque(maxlen=12))
    venue_age_e_history: dict[str, Deque[float]] = field(default_factory=dict)
    venue_age_a_history: dict[str, Deque[float]] = field(default_factory=dict)
    venue_delta_mid_history: dict[str, Deque[float]] = field(default_factory=dict)
    venue_reconnects: dict[str, int] = field(default_factory=dict)
    venue_cap_hits: dict[str, int] = field(default_factory=dict)
    venue_flash_until: dict[str, tuple[float, str]] = field(default_factory=dict)
    sys_age_e_max_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    sys_delta_mid_max_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    rx_rate_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    pnl_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    pos_history: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    run_fill_count: int = 0
    run_base_volume: float = 0.0
    run_notional_volume: float = 0.0
    run_fill_count_by_venue: dict[str, int] = field(default_factory=dict)
    run_base_volume_by_venue: dict[str, float] = field(default_factory=dict)
    run_notional_volume_by_venue: dict[str, float] = field(default_factory=dict)
    aster_guard_enabled: bool = False
    aster_guard_decision: str | None = None
    aster_guard_reason: str | None = None
    aster_guard_age_ms: int | None = None
    aster_guard_adverse_usd: float | None = None
    aster_guard_unrealised_usd: float | None = None
    aster_guard_cleanup_fee_usd: float | None = None
    aster_guard_allowed_orders: int = 0
    aster_guard_suppressed_orders: int = 0
    aster_guard_refresh_attempted: bool = False
    aster_guard_refresh_outcome: str | None = None
    aster_guard_refresh_latency_ms: int | None = None
    aster_guard_fresh_account_age_ms: int | None = None
    aster_guard_refresh_suppressed_reason: str | None = None
    rx_rate_ema: float | None = None
    balance_pnl: BalancePnlState = field(default_factory=BalancePnlState)
    balance_pnl_run_dir: str | None = None
    _balance_pnl_last_refresh_mono: float = 0.0
    _last_update_mono: float | None = None
    tick_count: int = 0
    prev_venue_status: dict[str, str] = field(default_factory=dict)
    venue_status_flips: dict[str, int] = field(default_factory=dict)
    venue_stale_ticks: dict[str, int] = field(default_factory=dict)
    global_cap_hits: int = 0
    frame_count: int = 0
    active_tab: int = _TAB_TAPE
    active_page: str = _PAGE_SIMPLE
    show_vn_map_until: float = 0.0
    layout_debug_printed: bool = False
    runner_status: RunnerStatus | None = None
    _seen_event_ids: Deque[str] = field(default_factory=deque)
    _seen_event_set: set[str] = field(default_factory=set)
    _seen_fill_volume_ids: set[str] = field(default_factory=set)

    def _remember_event_id(self, event_id: str) -> bool:
        if event_id in self._seen_event_set:
            return False
        if len(self._seen_event_ids) >= 4096:
            old = self._seen_event_ids.popleft()
            self._seen_event_set.discard(old)
        self._seen_event_ids.append(event_id)
        self._seen_event_set.add(event_id)
        return True

    def _append_tape(
        self,
        *,
        event_id: str,
        kind: str,
        text: str,
        venue_id: str | None,
        ts_ms: int | None,
    ) -> None:
        if not self._remember_event_id(event_id):
            return
        flash_kind = kind in {"buy", "sell", "cancel", "warn", "error", "kill"}
        flash_until = time.monotonic() + self.flash_ttl_s if flash_kind else 0.0
        self.tape.appendleft(
            TapeEvent(
                event_id=event_id,
                kind=kind,
                text=text,
                venue_id=venue_id,
                ts_ms=ts_ms,
                flash_until_mono=flash_until,
            )
        )
        if venue_id and flash_until > 0.0:
            self.venue_flash_until[venue_id] = (flash_until, kind)
        if kind in {"error", "kill"}:
            self.alerts.appendleft(text)

    def _append_history(
        self,
        history: dict[str, Deque[float]],
        venue_id: str,
        value: float | None,
    ) -> None:
        if value is None:
            return
        bucket = history.get(venue_id)
        if bucket is None:
            bucket = deque(maxlen=self.history_len)
            history[venue_id] = bucket
        bucket.append(value)

    def _append_series(self, bucket: Deque[float], value: float | None) -> None:
        if value is not None:
            bucket.append(float(value))

    def update(self, record: dict[str, Any]) -> None:
        self.last_record = record
        self.tick_count += 1
        venue_status = record.get("venue_status", [])
        venue_count = len(venue_status) if isinstance(venue_status, list) else 0
        self.venue_ids = parse_venue_ids(record, venue_count)
        venue_age_a = record.get("venue_age_ms", [])
        venue_age_e = record.get("venue_age_event_ms", [])
        venue_mid = record.get("venue_mid_usd", [])

        now_ms = None
        treasury = record.get("treasury_guidance")
        if isinstance(treasury, dict):
            now_ms = safe_int(treasury.get("as_of_ms"))

        _, delta_mid_map = compute_delta_mid_bps_map(
            self.venue_ids,
            venue_status if isinstance(venue_status, list) else [],
            venue_mid if isinstance(venue_mid, list) else [],
        )
        age_e_values: list[float] = []

        # Track per-venue stale%, status flips, and age trends.
        for idx, vid in enumerate(self.venue_ids):
            cur = (
                venue_status[idx]
                if isinstance(venue_status, list) and idx < len(venue_status)
                else None
            )
            age_a_val = (
                safe_float(venue_age_a[idx])
                if isinstance(venue_age_a, list) and idx < len(venue_age_a)
                else None
            )
            raw_age_e = (
                venue_age_e[idx]
                if isinstance(venue_age_e, list) and idx < len(venue_age_e)
                else None
            )
            age_e_val = safe_float(raw_age_e)
            if age_e_val is None:
                age_e_val = age_a_val
            if age_e_val is not None:
                age_e_values.append(age_e_val)
            self._append_history(self.venue_age_a_history, vid, age_a_val)
            self._append_history(self.venue_age_e_history, vid, age_e_val)
            self._append_history(
                self.venue_delta_mid_history,
                vid,
                delta_mid_map.get(vid),
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
                    if cur in {"Stale", "Disconnected", "Error", "Disabled"}:
                        tick = record.get("t", "n/a")
                        self._append_tape(
                            event_id=f"status:{vid}:{prev}->{cur}:{tick}",
                            kind="warn" if cur == "Stale" else "error",
                            text=f"{vid} status {prev}->{cur}",
                            venue_id=vid,
                            ts_ms=now_ms,
                        )
                self.prev_venue_status[vid] = cur

        sys_age_e_max = max(age_e_values) if age_e_values else None
        sys_delta_mid_max = max(
            (abs(value) for value in delta_mid_map.values() if value is not None),
            default=None,
        )
        self._append_series(self.sys_age_e_max_history, sys_age_e_max)
        self._append_series(self.sys_delta_mid_max_history, sys_delta_mid_max)

        now_mono = time.monotonic()
        if self._last_update_mono is not None:
            dt = now_mono - self._last_update_mono
            if dt > 0:
                inst_rate = min(250.0, 1.0 / dt)
                if self.rx_rate_ema is None:
                    self.rx_rate_ema = inst_rate
                else:
                    self.rx_rate_ema = (0.4 * inst_rate) + (0.6 * self.rx_rate_ema)
                self._append_series(self.rx_rate_history, self.rx_rate_ema)
        self._last_update_mono = now_mono

        pnl_value = _extract_pnl_usd(record)
        pos_value = _extract_net_pos_base(record)
        self._append_series(self.pnl_history, pnl_value)
        self._append_series(self.pos_history, pos_value)

        emergency_residual_fallback = record.get("emergency_residual_fallback")
        aster_guard = (
            emergency_residual_fallback.get("aster_residual_markout_guard")
            if isinstance(emergency_residual_fallback, dict)
            else None
        )
        if isinstance(aster_guard, dict):
            self.aster_guard_enabled = bool(aster_guard.get("enabled"))
            decision = aster_guard.get("decision")
            reason = aster_guard.get("reason")
            self.aster_guard_decision = decision if isinstance(decision, str) else None
            self.aster_guard_reason = reason if isinstance(reason, str) else None
            self.aster_guard_age_ms = safe_int(aster_guard.get("residual_age_ms"))
            self.aster_guard_adverse_usd = safe_float(
                aster_guard.get("adverse_markout_usd")
            )
            self.aster_guard_unrealised_usd = safe_float(
                aster_guard.get("residual_unrealised_usd")
            )
            self.aster_guard_cleanup_fee_usd = safe_float(
                aster_guard.get("cleanup_fee_estimate_usd")
            )
            self.aster_guard_allowed_orders = (
                safe_int(aster_guard.get("allowed_orders")) or 0
            )
            self.aster_guard_suppressed_orders = (
                safe_int(aster_guard.get("suppressed_orders")) or 0
            )
            self.aster_guard_refresh_attempted = bool(
                aster_guard.get("refresh_attempted")
            )
            refresh_outcome = aster_guard.get("refresh_outcome")
            self.aster_guard_refresh_outcome = (
                refresh_outcome if isinstance(refresh_outcome, str) else None
            )
            self.aster_guard_refresh_latency_ms = safe_int(
                aster_guard.get("refresh_latency_ms")
            )
            self.aster_guard_fresh_account_age_ms = safe_int(
                aster_guard.get("fresh_account_age_ms")
            )
            refresh_suppressed_reason = aster_guard.get("refresh_suppressed_reason")
            self.aster_guard_refresh_suppressed_reason = (
                refresh_suppressed_reason
                if isinstance(refresh_suppressed_reason, str)
                else None
            )
        else:
            self.aster_guard_enabled = False
            self.aster_guard_decision = None
            self.aster_guard_reason = None
            self.aster_guard_age_ms = None
            self.aster_guard_adverse_usd = None
            self.aster_guard_unrealised_usd = None
            self.aster_guard_cleanup_fee_usd = None
            self.aster_guard_allowed_orders = 0
            self.aster_guard_suppressed_orders = 0
            self.aster_guard_refresh_attempted = False
            self.aster_guard_refresh_outcome = None
            self.aster_guard_refresh_latency_ms = None
            self.aster_guard_fresh_account_age_ms = None
            self.aster_guard_refresh_suppressed_reason = None

        risk_events = record.get("risk_events", [])
        if isinstance(risk_events, list):
            for item in risk_events:
                if not isinstance(item, dict):
                    continue
                event_type = str(item.get("event_type", "risk_event"))
                event_l = event_type.lower()
                event_ts = safe_int(item.get("timestamp_ms")) or now_ms
                venue_idx = safe_int(item.get("venue_index"))
                venue_id = (
                    self.venue_ids[venue_idx]
                    if venue_idx is not None and 0 <= venue_idx < len(self.venue_ids)
                    else None
                )
                suffix = f" ({venue_id})" if venue_id else ""
                event_id = f"risk:{event_type}:{event_ts}:{venue_id}"
                if any(k in event_l for k in ("reconnect", "timeout", "watchdog")):
                    if venue_id:
                        self.venue_reconnects[venue_id] = (
                            self.venue_reconnects.get(venue_id, 0) + 1
                        )
                    self._append_tape(
                        event_id=event_id,
                        kind="error" if "timeout" in event_l else "warn",
                        text=f"{event_type}{suffix}",
                        venue_id=venue_id,
                        ts_ms=event_ts,
                    )
                elif "cap" in event_l:
                    self.global_cap_hits += 1
                    if venue_id:
                        self.venue_cap_hits[venue_id] = (
                            self.venue_cap_hits.get(venue_id, 0) + 1
                        )
                    self._append_tape(
                        event_id=event_id,
                        kind="warn",
                        text=f"{event_type}{suffix}",
                        venue_id=venue_id,
                        ts_ms=event_ts,
                    )
                elif "error" in event_l:
                    self._append_tape(
                        event_id=event_id,
                        kind="error",
                        text=f"{event_type}{suffix}",
                        venue_id=venue_id,
                        ts_ms=event_ts,
                    )

        fills = record.get("fills", [])
        if isinstance(fills, list):
            for fill_idx, fill in enumerate(fills):
                if not isinstance(fill, dict):
                    continue
                venue_id = fill.get("venue_id")
                if isinstance(venue_id, str):
                    fill_time = safe_int(fill.get("fill_time_ms"))
                    if fill_time is not None:
                        self.last_fill_ms[venue_id] = fill_time
                    size = fill.get("size")
                    price = fill.get("price")
                    side_raw = str(fill.get("side", "?")).upper()
                    side = (
                        "BUY"
                        if side_raw.startswith("B")
                        else "SELL"
                        if side_raw.startswith("S")
                        else side_raw
                    )
                    age = (
                        f"{int((now_ms - fill_time) / 1000)}s"
                        if now_ms and fill_time
                        else "n/a"
                    )
                    fill_text = f"{venue_id} {side} {size}@{price} age={age}"
                    fill_key = (
                        f"fill:{fill.get('id') or fill.get('order_id') or fill_idx}:"
                        f"{venue_id}:{side}:{size}:{price}:{fill_time}"
                    )
                    if fill_key not in self._seen_fill_volume_ids:
                        self._seen_fill_volume_ids.add(fill_key)
                        size_f = safe_float(size)
                        price_f = safe_float(price)
                        if size_f is not None:
                            abs_size = abs(size_f)
                            self.run_fill_count += 1
                            self.run_base_volume += abs_size
                            self.run_fill_count_by_venue[venue_id] = (
                                self.run_fill_count_by_venue.get(venue_id, 0) + 1
                            )
                            self.run_base_volume_by_venue[venue_id] = (
                                self.run_base_volume_by_venue.get(venue_id, 0.0)
                                + abs_size
                            )
                            if price_f is not None:
                                notional = abs_size * price_f
                                self.run_notional_volume += notional
                                self.run_notional_volume_by_venue[venue_id] = (
                                    self.run_notional_volume_by_venue.get(venue_id, 0.0)
                                    + notional
                                )
                    self.events.fills.appendleft(fill_text)
                    self._append_tape(
                        event_id=fill_key,
                        kind="buy" if side == "BUY" else "sell" if side == "SELL" else "warn",
                        text=fill_text,
                        venue_id=venue_id,
                        ts_ms=fill_time or now_ms,
                    )

        orders = record.get("orders", [])
        if isinstance(orders, list):
            for order_idx, order in enumerate(orders):
                if not isinstance(order, dict):
                    continue
                venue_id = order.get("venue_id", "n/a")
                if order.get("action") == "cancel" and order.get("status") == "ack":
                    reason = order.get("reason", "")
                    suffix = f" reason={reason}" if reason else ""
                    cancel_text = f"{venue_id} cancel{suffix}"
                    self.events.cancels.appendleft(cancel_text)
                    cancel_key = (
                        f"cancel:{order.get('order_id') or order_idx}:"
                        f"{venue_id}:{reason}:{record.get('t')}"
                    )
                    self._append_tape(
                        event_id=cancel_key,
                        kind="cancel",
                        text=cancel_text,
                        venue_id=venue_id if isinstance(venue_id, str) else None,
                        ts_ms=now_ms,
                    )
                reason = str(order.get("reason", "")).lower()
                if "cap" in reason:
                    self.global_cap_hits += 1
                    if isinstance(venue_id, str):
                        self.venue_cap_hits[venue_id] = (
                            self.venue_cap_hits.get(venue_id, 0) + 1
                        )

        if record.get("kill_switch"):
            reason = record.get("kill_reason", "unknown")
            tick = record.get("t", "n/a")
            kill_text = f"tick={tick} reason={reason}"
            self.events.kills.appendleft(kill_text)
            self._append_tape(
                event_id=f"kill:{tick}:{reason}",
                kind="kill",
                text=f"KILL {kill_text}",
                venue_id=None,
                ts_ms=now_ms,
            )

        if record.get("would_send_orders_truncated"):
            tick = record.get("t", "n/a")
            self.global_cap_hits += 1
            self._append_tape(
                event_id=f"would_send_orders_truncated:{tick}",
                kind="warn",
                text="would_send_orders truncated",
                venue_id=None,
                ts_ms=now_ms,
            )


def build_state(
    records: Iterable[dict[str, Any]],
    max_events: int,
    *,
    sort_mode: str = "agee",
    flash_ms: int = 650,
    page: str = _PAGE_SIMPLE,
) -> WatchState:
    state = WatchState(
        sort_mode=sort_mode,
        flash_ttl_s=max(0.05, flash_ms / 1000.0),
        active_page=page if page in {_PAGE_SIMPLE, _PAGE_EXPANDED} else _PAGE_SIMPLE,
    )
    state.events = EventLog(
        fills=deque(maxlen=max_events),
        cancels=deque(maxlen=max_events),
        kills=deque(maxlen=max_events),
    )
    state.tape = deque(maxlen=max_events)
    for record in records:
        state.update(record)
    return state


def rebuild_state_for_target(
    prior_state: WatchState,
    records: Iterable[dict[str, Any]],
    max_events: int,
    *,
    flash_ms: int,
) -> WatchState:
    state = build_state(
        records,
        max_events,
        sort_mode=prior_state.sort_mode,
        flash_ms=flash_ms,
        page=prior_state.active_page,
    )
    state.active_tab = prior_state.active_tab
    state.show_vn_map_until = prior_state.show_vn_map_until
    state.layout_debug_printed = prior_state.layout_debug_printed
    state.frame_count = prior_state.frame_count
    state.runner_status = prior_state.runner_status
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


# ── Rich rendering ───────────────────────────────────────────────────────────


_HEALTH_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


def fetch_health_detail_cached(
    base_url: str | None,
    ttl_s: float = 1.0,
) -> dict[str, Any] | None:
    if not base_url:
        return None
    now = time.monotonic()
    cached = _HEALTH_CACHE.get(base_url)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]
    detail = fetch_health_detail(base_url)
    _HEALTH_CACHE[base_url] = (now, detail)
    return detail


def _status_short(status: str) -> str:
    if status == "Healthy":
        return "OK"
    if status in {"Stale", "Disconnected", "Error"}:
        return "STALE"
    return "WARN"


def _status_style(status: str) -> str:
    if status == "Healthy":
        return "green"
    if status in {"Stale", "Disconnected", "Error"}:
        return "bold red"
    return "yellow"


def _status_rank(status: str) -> int:
    if status in {"Stale", "Disconnected", "Error"}:
        return 3
    if status == "Healthy":
        return 1
    return 2


def _sparkline(values: list[float], width: int) -> str:
    if width <= 0:
        return ""
    if not values:
        return "·" * min(width, 4)
    sample = values[-width:]
    lo = min(sample)
    hi = max(sample)
    if hi <= lo:
        return "▅" * len(sample)
    chars = "▁▂▃▄▅▆▇█"
    out: list[str] = []
    for value in sample:
        ratio = (value - lo) / (hi - lo)
        idx = max(0, min(len(chars) - 1, int(ratio * (len(chars) - 1))))
        out.append(chars[idx])
    return "".join(out)


def _trend_text(
    value_ms: float | None,
    history: list[float],
    *,
    compact: bool,
) -> Text:
    bar_width = 5 if compact else 7
    vmax = 5000.0
    ratio = 0.0
    if value_ms is not None:
        ratio = max(0.0, min(1.0, value_ms / vmax))
    filled = int(round(bar_width * ratio))
    gauge = ("█" * filled) + ("░" * (bar_width - filled))
    style = "green"
    if value_ms is not None and value_ms >= 5000:
        style = "bold red"
    elif value_ms is not None and value_ms >= 2000:
        style = "yellow"
    text = Text(gauge, style=style)
    if not compact:
        text.append(" ")
        text.append(_sparkline(history, width=8), style="cyan")
    return text


def _flash_style(kind: str, pulse: bool) -> str | None:
    strong = pulse
    if kind == "buy":
        return "black on bright_green" if strong else "black on green"
    if kind == "sell":
        return "white on bright_red" if strong else "white on red"
    if kind in {"cancel", "warn"}:
        return "black on bright_yellow" if strong else "black on yellow"
    if kind in {"error", "kill"}:
        return "white on bright_red" if strong else "bold white on red"
    return None


def _event_kind_style(kind: str) -> str:
    if kind == "buy":
        return "green"
    if kind == "sell":
        return "red"
    if kind in {"cancel", "warn"}:
        return "yellow"
    if kind in {"error", "kill"}:
        return "bold red"
    return "white"


def _event_kind_label(kind: str) -> str:
    if kind == "buy":
        return "BUY"
    if kind == "sell":
        return "SELL"
    if kind == "cancel":
        return "CXL"
    if kind == "warn":
        return "WARN"
    if kind == "error":
        return "ERR"
    if kind == "kill":
        return "KILL"
    return "EVT"


def _pick_number(sources: Iterable[dict[str, Any]], keys: Iterable[str]) -> float | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            number = safe_float(value)
            if number is not None:
                return number
    return None


def _mode_token(record: dict[str, Any]) -> str:
    mode = str(record.get("execution_mode", "n/a"))
    trade = str(record.get("trade_mode") or mode)
    token = trade.strip()
    return token.lower() if token else mode.lower()


def _is_shadow_mode(record: dict[str, Any]) -> bool:
    return "shadow" in _mode_token(record)


def _mode_display_token(
    record: dict[str, Any],
    runner_status: RunnerStatus | None,
) -> tuple[str, str]:
    token = _mode_token(record)
    if "shadow" in token:
        return ("SHA", "bold red")
    if runner_status is not None and runner_status.alive:
        label = token.upper() if token and token != "n/a" else "LIVE"
        return (label, "bold green")
    if token and token != "n/a":
        return ("SNAP", "bold yellow")
    return ("NO-RUN", "bold red")


def _simple_eage_bounds(
    state: WatchState,
    record: dict[str, Any],
) -> tuple[float | None, float | None]:
    venue_status = record.get("venue_status", [])
    venue_age_a = record.get("venue_age_ms", [])
    venue_age_e = record.get("venue_age_event_ms", [])
    venue_count = len(state.venue_ids)
    if venue_count <= 0:
        venue_count = len(venue_status) if isinstance(venue_status, list) else 0
    if venue_count <= 0:
        venue_count = max(
            len(venue_age_a) if isinstance(venue_age_a, list) else 0,
            len(venue_age_e) if isinstance(venue_age_e, list) else 0,
        )

    values: list[float] = []
    for idx in range(venue_count):
        status = (
            str(venue_status[idx])
            if isinstance(venue_status, list) and idx < len(venue_status)
            else "Unknown"
        )
        if status != "Healthy":
            continue
        raw_age_e = (
            venue_age_e[idx]
            if isinstance(venue_age_e, list) and idx < len(venue_age_e)
            else None
        )
        age_e = safe_float(raw_age_e)
        if age_e is None:
            age_e = (
                safe_float(venue_age_a[idx])
                if isinstance(venue_age_a, list) and idx < len(venue_age_a)
                else None
            )
        if age_e is None or age_e < 0:
            continue
        values.append(age_e)

    if not values:
        return None, None
    return min(values), max(values)


def _format_simple_age_ms(value: float | None) -> str:
    if value is None:
        return "---"
    ivalue = max(0, int(round(value)))
    if ivalue < 1000:
        return f"{ivalue:03d}"
    return str(ivalue)


def _simple_tape_tag(kind: str) -> tuple[str, str]:
    if kind == "buy":
        return "BUY", "green"
    if kind == "sell":
        return "SELL", "red"
    if kind == "cancel":
        return "CXL", "yellow"
    if kind in {"error", "kill"}:
        return "WARN", "bold red"
    if kind == "warn":
        return "WARN", "yellow"
    return "INFO", "dim"


def _compact_tape_payload(text: str, width: int) -> str:
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        normalized = "-"
    if width <= 0:
        return normalized
    if len(normalized) <= width:
        return normalized
    if width <= 3:
        return normalized[:width]
    return normalized[: width - 3] + "..."


QUOTE_FOOTER = (
    "'all history present in that visage, the child the father of the man' - "
    "McCarthy (c.1985; 1933 - 2023 AD)"
)


def _extract_net_pos_usd(record: dict[str, Any]) -> float | None:
    return _pick_number(
        [record],
        (
            "net_position_usd",
            "position_usd",
            "dollar_delta_usd",
        ),
    )


def _extract_net_pos_base(record: dict[str, Any]) -> float | None:
    return _pick_number(
        [record],
        (
            "q_global_tao",
            "net_position_tao",
            "position_tao",
        ),
    )


def _extract_max_pos_cap_usd(record: dict[str, Any]) -> float | None:
    return _pick_number(
        [record],
        (
            "max_pos_cap_usd",
            "max_position_cap_usd",
            "max_position_usd",
            "position_cap_usd",
            "max_abs_position_usd",
            "delta_limit_usd",
        ),
    )


def _extract_pnl_usd(record: dict[str, Any]) -> float | None:
    return _pick_number(
        [record],
        (
            "pnl_total",
            "pnl_total_usd",
            "pnl_usd",
            "pnl_unrealised",
            "pnl_unrealized",
            "unrealized_pnl_usd",
            "pnl_realised",
            "pnl_realized",
        ),
    )


def _format_signed_dollars(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) < 1.0:
        return f"{value:+.4f}"
    return f"{value:+,.2f}"


def _format_balance_pnl_short(state: BalancePnlState) -> str:
    if state.status == "available":
        return _format_signed_dollars(state.delta_usd)
    if state.status == "pending_post":
        return "pending"
    if state.status == "unreadable":
        return "error"
    return "n/a"


def _balance_pnl_style(state: BalancePnlState) -> str:
    if state.status == "available":
        if state.delta_usd is None:
            return "dim"
        if state.delta_usd > 0:
            return "green"
        if state.delta_usd < 0:
            return "red"
        return "white"
    if state.status == "pending_post":
        return "yellow"
    if state.status == "unreadable":
        return "bold red"
    return "dim"


def _delta_pct_style(value: float | None) -> str:
    if value is None:
        return "dim"
    if value >= 75.0:
        return "bold red"
    if value >= 35.0:
        return "yellow"
    return "green"


def _compute_gate(record: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if record.get("kill_switch"):
        reasons.append("kill")
    regime = str(record.get("risk_regime", "")).lower()
    if regime in {"emergency", "hardstop"}:
        reasons.append(regime)
    healthy = safe_int(record.get("healthy_venues_used_count"))
    if healthy is not None and healthy <= 0:
        reasons.append("healthy=0")
    return (len(reasons) == 0, reasons)


def _build_venue_rows(state: WatchState, record: dict[str, Any]) -> list[dict[str, Any]]:
    venue_status = record.get("venue_status", [])
    venue_mid = record.get("venue_mid_usd", [])
    venue_spread = record.get("venue_spread_usd", [])
    venue_age_a = record.get("venue_age_ms", [])
    venue_age_e = record.get("venue_age_event_ms", [])
    _, delta_mid_map = compute_delta_mid_bps_map(
        state.venue_ids,
        venue_status if isinstance(venue_status, list) else [],
        venue_mid if isinstance(venue_mid, list) else [],
    )
    rows: list[dict[str, Any]] = []
    for idx, venue_id in enumerate(state.venue_ids):
        status = (
            venue_status[idx]
            if isinstance(venue_status, list) and idx < len(venue_status)
            else "Unknown"
        )
        mid = (
            safe_float(venue_mid[idx])
            if isinstance(venue_mid, list) and idx < len(venue_mid)
            else None
        )
        spread_usd = (
            safe_float(venue_spread[idx])
            if isinstance(venue_spread, list) and idx < len(venue_spread)
            else None
        )
        age_a = (
            safe_float(venue_age_a[idx])
            if isinstance(venue_age_a, list) and idx < len(venue_age_a)
            else None
        )
        raw_age_e = (
            venue_age_e[idx]
            if isinstance(venue_age_e, list) and idx < len(venue_age_e)
            else None
        )
        age_e = safe_float(raw_age_e)
        if age_e is None:
            age_e = age_a
        spread_bps = None
        if mid and spread_usd is not None and mid != 0:
            spread_bps = (spread_usd / mid) * 10000.0
        stale_ticks = state.venue_stale_ticks.get(venue_id, 0)
        stale_pct = (100.0 * stale_ticks / state.tick_count) if state.tick_count > 0 else 0.0
        rows.append(
            {
                "venue": venue_id,
                "status": status,
                "status_short": _status_short(str(status)),
                "age_e": age_e,
                "age_a": age_a,
                "spread_bps": spread_bps,
                "mid": mid,
                "delta_mid_bps": delta_mid_map.get(venue_id),
                "flips": state.venue_status_flips.get(venue_id, 0),
                "stale_pct": stale_pct,
                "recon": state.venue_reconnects.get(venue_id, 0),
                "cap": state.venue_cap_hits.get(venue_id, 0),
                "history_e": list(state.venue_age_e_history.get(venue_id, [])),
                "history_delta_mid": list(state.venue_delta_mid_history.get(venue_id, [])),
            }
        )

    if state.sort_mode == "stale":
        rows.sort(
            key=lambda row: (
                row["stale_pct"],
                _status_rank(str(row["status"])),
                row["age_e"] if row["age_e"] is not None else -1.0,
            ),
            reverse=True,
        )
    elif state.sort_mode == "venue":
        rows.sort(key=lambda row: str(row["venue"]))
    else:
        rows.sort(
            key=lambda row: (
                row["age_e"] if row["age_e"] is not None else -1.0,
                row["stale_pct"],
                _status_rank(str(row["status"])),
            ),
            reverse=True,
        )
    return rows


def _quantize_for_key(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _build_frame_key(state: WatchState, now_mono: float) -> tuple[Any, ...]:
    record = state.last_record or {}
    if state.active_page == _PAGE_SIMPLE:
        min_age_e, max_age_e = _simple_eage_bounds(state, record)
        tape_events = list(state.tape)[:8]
        return (
            _PAGE_SIMPLE,
            record.get("t"),
            _is_shadow_mode(record),
            int(datetime.now(timezone.utc).timestamp() // 60),
            int(min_age_e) if min_age_e is not None else None,
            int(max_age_e) if max_age_e is not None else None,
            tuple(
                (
                    event.event_id,
                    event.ts_ms,
                    now_mono < event.flash_until_mono,
                )
                for event in tape_events
            ),
            _quantize_for_key(_extract_net_pos_usd(record), 0),
            _quantize_for_key(_extract_max_pos_cap_usd(record), 0),
            _quantize_for_key(_extract_pnl_usd(record), 4),
            state.balance_pnl.status,
            _quantize_for_key(state.balance_pnl.delta_usd, 4),
        )

    venue_status = record.get("venue_status", [])
    venue_mid = record.get("venue_mid_usd", [])
    venue_age_a = record.get("venue_age_ms", [])
    venue_age_e = record.get("venue_age_event_ms", [])
    _, delta_mid_map = compute_delta_mid_bps_map(
        state.venue_ids,
        venue_status if isinstance(venue_status, list) else [],
        venue_mid if isinstance(venue_mid, list) else [],
    )
    rows: list[dict[str, Any]] = []
    for idx, venue_id in enumerate(state.venue_ids):
        status = (
            str(venue_status[idx])
            if isinstance(venue_status, list) and idx < len(venue_status)
            else "Unknown"
        )
        age_a = (
            safe_float(venue_age_a[idx])
            if isinstance(venue_age_a, list) and idx < len(venue_age_a)
            else None
        )
        raw_age_e = (
            venue_age_e[idx]
            if isinstance(venue_age_e, list) and idx < len(venue_age_e)
            else None
        )
        age_e = safe_float(raw_age_e)
        if age_e is None:
            age_e = age_a
        mid = (
            safe_float(venue_mid[idx])
            if isinstance(venue_mid, list) and idx < len(venue_mid)
            else None
        )
        stale_ticks = state.venue_stale_ticks.get(venue_id, 0)
        stale_pct = (100.0 * stale_ticks / state.tick_count) if state.tick_count > 0 else 0.0
        rows.append(
            {
                "venue": venue_id,
                "status": status,
                "age_e": age_e,
                "stale_pct": stale_pct,
                "row": (
                    venue_id,
                    status,
                    int(age_e) if age_e is not None else None,
                    int(age_a) if age_a is not None else None,
                    _quantize_for_key(mid, 4),
                    _quantize_for_key(delta_mid_map.get(venue_id), 2),
                    _quantize_for_key(stale_pct, 1),
                    int(state.venue_reconnects.get(venue_id, 0)),
                    int(state.venue_cap_hits.get(venue_id, 0)),
                ),
            }
        )
    if state.sort_mode == "stale":
        rows.sort(
            key=lambda row: (
                row["stale_pct"],
                _status_rank(str(row["status"])),
                row["age_e"] if row["age_e"] is not None else -1.0,
            ),
            reverse=True,
        )
    elif state.sort_mode == "venue":
        rows.sort(key=lambda row: str(row["venue"]))
    else:
        rows.sort(
            key=lambda row: (
                row["age_e"] if row["age_e"] is not None else -1.0,
                row["stale_pct"],
                _status_rank(str(row["status"])),
            ),
            reverse=True,
        )

    latest_tape = state.tape[0] if state.tape else None
    return (
        _PAGE_EXPANDED,
        record.get("t"),
        tuple(row["row"] for row in rows),
        latest_tape.event_id if latest_tape else None,
        latest_tape.ts_ms if latest_tape else None,
        state.active_tab,
        now_mono < state.show_vn_map_until,
        _quantize_for_key(_extract_pnl_usd(record), 4),
        state.balance_pnl.status,
        _quantize_for_key(state.balance_pnl.delta_usd, 4),
    )


def _format_ms_short(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value)}ms"


def _format_bps_short(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}bps"


def _format_rate_short(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def _format_pnl_short(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value/1000.0:+.1f}k"
    if abs(value) < 1.0:
        return f"{value:+.4f}"
    return f"{value:+.2f}"


def _format_pos_short(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}"


def _format_base_volume_short(value: float) -> str:
    if value >= 100:
        return f"{value:,.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _format_notional_short(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000.0:.2f}m"
    if value >= 1000:
        return f"${value / 1000.0:.2f}k"
    return f"${value:.2f}"


def _format_aster_guard_short(state: WatchState) -> str | None:
    if not state.aster_guard_enabled:
        return None
    decision = state.aster_guard_decision or "armed"
    reason = state.aster_guard_reason or "no_decision"
    adverse = _format_pnl_short(state.aster_guard_adverse_usd)
    age = _format_ms_short(float(state.aster_guard_age_ms) if state.aster_guard_age_ms is not None else None)
    refresh = ""
    if state.aster_guard_refresh_outcome:
        refresh_latency = _format_ms_short(
            float(state.aster_guard_refresh_latency_ms)
            if state.aster_guard_refresh_latency_ms is not None
            else None
        )
        refresh_age = _format_ms_short(
            float(state.aster_guard_fresh_account_age_ms)
            if state.aster_guard_fresh_account_age_ms is not None
            else None
        )
        refresh = (
            f" refresh={state.aster_guard_refresh_outcome}"
            f"/{refresh_latency} acct_age={refresh_age}"
        )
    return (
        f"{decision}:{reason} adv={adverse} age={age} "
        f"a/s={state.aster_guard_allowed_orders}/{state.aster_guard_suppressed_orders}"
        f"{refresh}"
    )


def _format_mid_cell(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def _format_cell_age(value: float | None) -> Text:
    if value is None:
        return Text("—", style="dim")
    style = "white"
    if value >= 5000:
        style = "bold red"
    elif value >= 2000:
        style = "yellow"
    return Text(f"{int(value)}", style=style)


def _format_cell_bps(value: float | None, *, signed: bool = False) -> Text:
    if value is None:
        return Text("—", style="dim")
    style = "white"
    abs_v = abs(value)
    if abs_v >= 20:
        style = "bold red"
    elif abs_v >= 5:
        style = "yellow"
    text = f"{value:+.2f}" if signed else f"{value:.2f}"
    return Text(text, style=style)


def _format_cell_stale(pct: float) -> Text:
    style = "green"
    if pct >= 5.0:
        style = "bold red"
    elif pct >= 1.0:
        style = "yellow"
    return Text(f"{pct:4.1f}%", style=style)


def _tabs_line(active_tab: int) -> Text:
    text = Text()
    labels = ("TAPE", "KEYS", "ALERTS")
    for idx, label in enumerate(labels):
        if idx > 0:
            text.append(" / ", style="dim")
        style = "underline bold cyan" if idx == active_tab else "dim"
        text.append(label, style=style)
    return text


def _compute_layout_rows(term_height: int) -> dict[str, int]:
    rows = {
        "header_rows": 3,
        "legend_rows": 1,
        "graph_rows": 3,
        "venues_rows": 9,  # panel + status strip + header + 5 venues
        "tabs_rows": 4,
    }

    fixed = (
        rows["header_rows"]
        + rows["legend_rows"]
        + rows["graph_rows"]
        + rows["venues_rows"]
    )
    if term_height > fixed:
        rows["tabs_rows"] = max(4, term_height - fixed)
    else:
        deficit = (fixed + rows["tabs_rows"]) - term_height
        for key, minimum in (
            ("tabs_rows", 2),
            ("graph_rows", 2),
            ("legend_rows", 0),
            ("tabs_rows", 1),
            ("graph_rows", 1),
            ("venues_rows", 7),
            ("header_rows", 2),
            ("venues_rows", 5),
        ):
            if deficit <= 0:
                break
            value = rows[key]
            reducible = max(0, value - minimum)
            take = min(deficit, reducible)
            rows[key] = value - take
            deficit -= take
        if deficit > 0:
            rows["tabs_rows"] = max(1, rows["tabs_rows"] - deficit)

    out = {
        "header": max(1, rows["header_rows"]),
        "legend": max(0, rows["legend_rows"]),
        "graph": max(1, rows["graph_rows"]),
        "venues": max(1, rows["venues_rows"]),
        "tabs": max(1, rows["tabs_rows"]),
    }
    while (
        out["header"] + out["legend"] + out["graph"] + out["venues"] + out["tabs"]
    ) > max(1, term_height):
        for key in ("tabs", "graph", "legend", "header", "venues"):
            minimum = 0 if key == "legend" else 1
            if out[key] > minimum:
                out[key] -= 1
                break
        else:
            break
    return out


def render_frame_expanded(
    state: WatchState,
    max_events: int,
    *,
    term_width: int,
    term_height: int,
    config_dir: str | None = None,
    health_url: str | None = None,
    layout_debug: bool = False,
    ui_mode: str | None = None,
    render_ms: int | None = None,
) -> Any:
    state.frame_count += 1
    record = state.last_record or {}
    health = fetch_health_detail_cached(health_url) if health_url else None
    now_mono = time.monotonic()
    pulse = (state.frame_count % 2) == 0

    treasury = record.get("treasury_guidance")
    now_ms = None
    if isinstance(treasury, dict):
        now_ms = safe_int(treasury.get("as_of_ms"))

    rows = _build_venue_rows(state, record)
    gate_pass, gate_reasons = _compute_gate(record)
    run_id = str(record.get("config_version_id") or "n/a")
    run_limit = 20 if term_width < 120 else 28
    if len(run_id) > run_limit:
        run_id = run_id[: run_limit - 1] + "…"
    mode_token, mode_style = _mode_display_token(record, state.runner_status)
    in_shadow = _is_shadow_mode(record)
    gate_text = "PASS" if gate_pass else f"FAIL({','.join(gate_reasons)})"
    tick_value = record.get("t", "n/a")
    time_text = short_ts(now_ms) if now_ms else "n/a"
    wall_time_text = wall_clock_utc()
    src_age_text, src_age_style = source_age_label(state, now_mono)
    runner_text, runner_style = runner_status_label(state)
    heartbeat = heartbeat_glyph(state.frame_count)
    recon_total = sum(state.venue_reconnects.values())
    cap_hits = state.global_cap_hits

    ws_audit_raw = record.get("ws_audit")
    if ws_audit_raw is None:
        ws_audit_raw = os.environ.get("PARAPHINA_WS_AUDIT")
    ws_audit_text = "UNK"
    if isinstance(ws_audit_raw, bool):
        ws_audit_text = "ON" if ws_audit_raw else "OFF"
    elif isinstance(ws_audit_raw, str):
        ws_audit_text = "ON" if ws_audit_raw.strip().lower() in {"1", "true", "yes", "on"} else "OFF"

    worst_age_e = max((row for row in rows if row["age_e"] is not None), key=lambda row: row["age_e"], default=None)
    worst_age_a = max((row for row in rows if row["age_a"] is not None), key=lambda row: row["age_a"], default=None)
    worst_dmid = max(
        (row for row in rows if row["delta_mid_bps"] is not None),
        key=lambda row: abs(float(row["delta_mid_bps"])),
        default=None,
    )

    layout_rows = _compute_layout_rows(term_height)
    if layout_debug and not state.layout_debug_printed:
        print(
            (
                f"[layout] h={term_height} header={layout_rows['header']} "
                f"legend={layout_rows['legend']} graph={layout_rows['graph']} "
                f"venues={layout_rows['venues']} tabs={layout_rows['tabs']}"
            ),
            file=sys.stderr,
        )
        state.layout_debug_printed = True

    header_grid = Table.grid(expand=True)
    header_grid.add_column(ratio=1, justify="left")
    header_grid.add_column(justify="right", no_wrap=True)
    left_header = Text("paraphina v1.1", style="bold cyan")
    left_header.append("  watch", style="bold white")
    right_header = Text()
    right_header.append(mode_token, style=mode_style)
    right_header.append("  GATE ", style="dim")
    right_header.append(gate_text, style="bold green" if gate_pass else "bold red")
    right_header.append(f"  t {tick_value}", style="white")
    right_header.append(f"  rec {time_text}", style="dim")
    right_header.append(f"  {src_age_text}", style=src_age_style)
    right_header.append(f"  {runner_text}", style=runner_style)
    right_header.append(f"  now {wall_time_text}", style="white")
    right_header.append(f"  {heartbeat}", style="cyan")
    right_header.append(f"  {run_id}", style="dim")
    header_grid.add_row(left_header, right_header)
    header_panel = Panel(
        header_grid,
        box=box.SQUARE,
        border_style="cyan",
        padding=(0, 1),
    )

    legend_line = Text(
        "ageE(ts→rx)  ageA(rx→publish)  stale%=pct beyond SLA  flips=mid flips  Δmid(bps)=mid−median(mid_all)  bPNL=balance pre/post",
        style="dim",
    )

    spark_w = 8 if term_width >= 140 else 6 if term_width >= 110 else 5
    sys_age_e_now = worst_age_e["age_e"] if worst_age_e else None
    sys_dmid_now = abs(float(worst_dmid["delta_mid_bps"])) if worst_dmid else None
    rx_now = state.rx_rate_history[-1] if state.rx_rate_history else None
    pos_now = state.pos_history[-1] if state.pos_history else None
    flat_hist = [0.0] * max(4, spark_w)

    graph_segments: list[Text] = []
    seg1 = Text("SYS ageE max ", style="dim")
    seg1.append(_format_ms_short(sys_age_e_now), style="white")
    seg1.append(" ")
    seg1.append(_sparkline(list(state.sys_age_e_max_history), width=spark_w), style="cyan")
    graph_segments.append(seg1)

    seg2 = Text("SYS Δmid max ", style="dim")
    seg2.append(_format_bps_short(sys_dmid_now), style="white")
    seg2.append(" ")
    seg2.append(_sparkline(list(state.sys_delta_mid_max_history), width=spark_w), style="cyan")
    graph_segments.append(seg2)

    seg3 = Text("RX/s ", style="dim")
    seg3.append(_format_rate_short(rx_now), style="white")
    seg3.append(" ")
    seg3.append(_sparkline(list(state.rx_rate_history), width=spark_w), style="cyan")
    graph_segments.append(seg3)

    seg4 = Text("bPNL ", style="dim")
    if in_shadow:
        seg4.append("— (shadow)", style="dim")
        seg4.append(" ")
        seg4.append(_sparkline(flat_hist, width=spark_w), style="dim")
    elif state.balance_pnl.status in {"available", "pending_post", "unreadable"}:
        seg4.append(
            _format_balance_pnl_short(state.balance_pnl),
            style=_balance_pnl_style(state.balance_pnl),
        )
        seg4.append(" ")
        seg4.append("balances", style="dim")
    else:
        pnl_now = state.pnl_history[-1] if state.pnl_history else None
        seg4.append(_format_pnl_short(pnl_now), style="white")
        seg4.append(" ")
        seg4.append("tPNL", style="dim")
    graph_segments.append(seg4)

    seg5 = Text("POSb ", style="dim")
    if in_shadow:
        seg5.append("— (shadow)", style="dim")
        seg5.append(" ")
        seg5.append(_sparkline(flat_hist, width=spark_w), style="dim")
    else:
        seg5.append(_format_pos_short(pos_now), style="white")
        seg5.append(" ")
        seg5.append(_sparkline(list(state.pos_history), width=spark_w), style="cyan")
    graph_segments.append(seg5)

    seg6 = Text("sVOL ", style="dim")
    seg6.append(str(state.run_fill_count), style="white")
    seg6.append("f ", style="dim")
    seg6.append(_format_base_volume_short(state.run_base_volume), style="white")
    seg6.append("e ", style="dim")
    seg6.append(_format_notional_short(state.run_notional_volume), style="white")
    graph_segments.append(seg6)

    graph_line = Text()
    for idx, segment in enumerate(graph_segments):
        if idx > 0:
            graph_line.append(" │ ", style="dim")
        graph_line.append_text(segment)
    graph_panel = Panel(
        graph_line,
        box=box.SQUARE,
        border_style="cyan",
        padding=(0, 1),
    )

    status_strip = Text()
    status_strip.append("gate ", style="dim")
    status_strip.append("PASS" if gate_pass else "FAIL", style="green" if gate_pass else "bold red")
    status_strip.append("  ws ", style="dim")
    status_strip.append(ws_audit_text, style="green" if ws_audit_text == "ON" else "yellow")
    status_strip.append("  worstE ", style="dim")
    if worst_age_e:
        status_strip.append(f"{int(worst_age_e['age_e'])}ms@{venue_code(str(worst_age_e['venue']))}", style="white")
    else:
        status_strip.append("—", style="dim")
    status_strip.append("  worstA ", style="dim")
    if worst_age_a:
        status_strip.append(f"{int(worst_age_a['age_a'])}ms@{venue_code(str(worst_age_a['venue']))}", style="white")
    else:
        status_strip.append("—", style="dim")
    status_strip.append("  worstΔ ", style="dim")
    if worst_dmid:
        status_strip.append(
            f"{float(worst_dmid['delta_mid_bps']):+.1f}bps@{venue_code(str(worst_dmid['venue']))}",
            style="white",
        )
    else:
        status_strip.append("—", style="dim")
    status_strip.append(f"  cap {cap_hits}", style="dim")
    status_strip.append(f"  recon {recon_total}", style="dim")
    aster_guard_text = _format_aster_guard_short(state)
    if aster_guard_text:
        guard_style = (
            "green"
            if state.aster_guard_decision == "allow"
            else "yellow"
            if state.aster_guard_decision == "suppress"
            else "dim"
        )
        status_strip.append("  ag ", style="dim")
        status_strip.append(aster_guard_text, style=guard_style)
    status_strip.append(f"  alerts {len(state.alerts)}", style="dim")
    if ui_mode:
        status_strip.append(f"  ui={ui_mode}", style="dim")
    if render_ms is not None:
        status_strip.append(f"  render={int(render_ms)}ms", style="dim")
    status_strip.append("  ?=VN map", style="dim")

    venue_table = Table(
        box=None,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        pad_edge=False,
    )
    venue_table.add_column("VN", no_wrap=True, width=2, justify="left")
    venue_table.add_column("st", no_wrap=True, width=2, justify="left")
    venue_table.add_column("ageE", no_wrap=True, justify="right")
    venue_table.add_column("ageA", no_wrap=True, justify="right")
    venue_table.add_column("spr(bps)", no_wrap=True, justify="right")
    venue_table.add_column("mid", no_wrap=True, justify="right")
    venue_table.add_column("Δmid", no_wrap=True, justify="right")
    venue_table.add_column("flips", no_wrap=True, justify="right")
    venue_table.add_column("stale%", no_wrap=True, justify="right")
    venue_table.add_column("recon", no_wrap=True, justify="right")
    venue_table.add_column("cap", no_wrap=True, justify="right")
    trend_w = 7 if term_width >= 130 else 5
    venue_table.add_column("trendE", no_wrap=True, justify="left")
    venue_table.add_column("trendΔmid", no_wrap=True, justify="left")

    display_rows = list(rows[:5])
    while len(display_rows) < 5:
        display_rows.append(
            {
                "venue": "--",
                "status": "Unknown",
                "status_short": "--",
                "age_e": None,
                "age_a": None,
                "spread_bps": None,
                "mid": None,
                "delta_mid_bps": None,
                "flips": 0,
                "stale_pct": 0.0,
                "recon": 0,
                "cap": 0,
                "history_e": [],
                "history_delta_mid": [],
            }
        )

    for row in display_rows:
        status = str(row["status"])
        vn = venue_code(str(row["venue"])) if row["venue"] != "--" else "--"
        vn_style = "bold bright_blue"
        flash_meta = state.venue_flash_until.get(str(row["venue"]))
        if flash_meta and now_mono < flash_meta[0]:
            vn_style = _flash_style(flash_meta[1], pulse) or vn_style
        trend_e_text = Text(_sparkline(list(row["history_e"]), width=trend_w), style="cyan")
        trend_d_values = [
            abs(float(value))
            for value in row["history_delta_mid"]
            if isinstance(value, (int, float))
        ]
        trend_d_text = Text(_sparkline(trend_d_values, width=trend_w), style="cyan")
        venue_table.add_row(
            Text(vn, style=vn_style),
            Text(row["status_short"], style=_status_style(status)),
            _format_cell_age(row["age_e"]),
            _format_cell_age(row["age_a"]),
            _format_cell_bps(row["spread_bps"]),
            Text(_format_mid_cell(row["mid"]), style="white" if row["mid"] is not None else "dim"),
            _format_cell_bps(row["delta_mid_bps"], signed=True),
            Text(str(int(row["flips"])), style="white"),
            _format_cell_stale(float(row["stale_pct"])),
            Text(str(int(row["recon"])), style="white"),
            Text(str(int(row["cap"])), style="white"),
            trend_e_text,
            trend_d_text,
        )

    venues_content = [status_strip]
    if not rows:
        venues_content.append(Text("waiting for telemetry…", style="dim"))
    venues_content.append(venue_table)
    venues_panel = Panel(
        Group(*venues_content),
        title="VENUES (worst-first)",
        box=box.SQUARE,
        border_style="cyan",
        padding=(0, 1),
    )

    active_tab = state.active_tab if state.active_tab in {_TAB_TAPE, _TAB_KEYS, _TAB_ALERTS} else _TAB_TAPE
    tabs_budget = max(1, layout_rows["tabs"] - 3)
    show_map = now_mono < state.show_vn_map_until
    tabs_lines: list[Text] = []
    if show_map and tabs_budget > 0:
        tabs_lines.append(
            Text("VN map: HL=hyperliquid AS=aster EX=extended PA=paradex LG=lighter", style="dim")
        )
    remaining = max(1, tabs_budget - len(tabs_lines))
    if active_tab == _TAB_TAPE:
        tape_limit = max(1, min(max_events, remaining))
        tape_events = list(state.tape)[:tape_limit]
        if tape_events:
            for event in tape_events:
                ts = short_ts(event.ts_ms) if event.ts_ms else "--:--:--"
                line = Text(f"{ts} ", style="dim")
                label = _event_kind_label(event.kind)
                if now_mono < event.flash_until_mono:
                    style = _flash_style(event.kind, pulse) or _event_kind_style(event.kind)
                    line.append(f"{label:<5} {event.text}", style=style)
                else:
                    line.append(f"{label:<5}", style=_event_kind_style(event.kind))
                    line.append(f" {event.text}", style="white")
                tabs_lines.append(line)
        else:
            tabs_lines.append(Text("waiting for telemetry…", style="dim"))
    elif active_tab == _TAB_KEYS:
        key_lines = [
            Text("1/2/3 or t/k/a: switch tabs", style="white"),
            Text("←/→: cycle tabs", style="white"),
            Text("?: show VN map", style="white"),
            Text("Ctrl-C: exit", style="white"),
        ]
        tabs_lines.extend(key_lines[:remaining])
    else:
        alert_items = list(state.alerts)[:remaining]
        if alert_items:
            for item in alert_items:
                tabs_lines.append(Text(item, style="bold red"))
        else:
            tabs_lines.append(Text("no active alerts", style="dim"))
    tabs_lines = tabs_lines[: max(1, tabs_budget)]

    tabs_grid = Table.grid(expand=True)
    tabs_grid.add_column(ratio=1)
    tabs_grid.add_row(_tabs_line(active_tab))
    for line in tabs_lines:
        tabs_grid.add_row(line)
    tabs_panel = Panel(
        tabs_grid,
        box=box.SQUARE,
        border_style="cyan",
        padding=(0, 1),
    )

    root = Layout()
    sections: list[Layout] = [Layout(header_panel, size=layout_rows["header"])]
    if layout_rows["legend"] > 0:
        sections.append(Layout(legend_line, size=layout_rows["legend"]))
    sections.extend(
        [
            Layout(graph_panel, size=layout_rows["graph"]),
            Layout(venues_panel, size=layout_rows["venues"]),
            Layout(tabs_panel, size=layout_rows["tabs"]),
        ]
    )
    root.split_column(*sections)
    return root


def render_frame_simple(
    state: WatchState,
    max_events: int,
    *,
    term_width: int,
    term_height: int,
    config_dir: str | None = None,
    health_url: str | None = None,
    layout_debug: bool = False,
    ui_mode: str | None = None,
    render_ms: int | None = None,
) -> Any:
    del max_events, config_dir, health_url, layout_debug, ui_mode, render_ms
    state.frame_count += 1
    record = state.last_record or {}
    now_mono = time.monotonic()
    pulse = (state.frame_count % 2) == 0

    in_shadow = _is_shadow_mode(record)
    mode_label, mode_style = _mode_display_token(record, state.runner_status)
    min_age_e, max_age_e = _simple_eage_bounds(state, record)
    tick_value = record.get("t", "n/a")
    wall_time_text = wall_clock_utc()
    src_age_text, src_age_style = source_age_label(state, now_mono)
    runner_text, runner_style = runner_status_label(state)
    heartbeat = heartbeat_glyph(state.frame_count)

    header = Text("www.paraphina.com // 1.89 // ")
    header.append(mode_label, style=mode_style)
    header.append(" // ")
    header.append(
        _format_simple_age_ms(min_age_e),
        style="white" if min_age_e is not None else "dim",
    )
    header.append(" // ")
    header.append(
        _format_simple_age_ms(max_age_e),
        style="white" if max_age_e is not None else "dim",
    )
    header.append(" // ")
    header.append(wall_time_text, style="white")
    header.append(" // ")
    header.append(src_age_text, style=src_age_style)
    header.append(" // ")
    header.append(runner_text, style=runner_style)
    header.append(" \\ ")
    header.append(str(tick_value), style="white")
    header.append(f" {heartbeat}", style="cyan")

    legend = Text(
        "EAGE (ts-rx)  AAGE (rx-publish)  bPNL balances pre/post  POSb base  Δ$ telemetry  sVOL session fills/base/notional",
        style="dim italic",
    )

    tape_prefix = "tape / "
    tape_indent = " " * len(tape_prefix)
    tape_payload_width = max(16, term_width - len(tape_prefix) - 18)
    tape_events = list(state.tape)[:8]
    tape_lines: list[Text] = []
    if not tape_events:
        if state.last_record is None:
            line = Text("tape / --:--:-- ", style="dim")
            line.append("INFO", style="white")
            if state.runner_status is not None and not state.runner_status.alive:
                line.append(" runner is not live", style="bold red")
                line.append(f" ({runner_text})", style=runner_style)
            else:
                line.append(" waiting for telemetry...", style="dim")
                line.append(f" ({runner_text})", style=runner_style)
            tape_lines.append(line)
        else:
            src_age_text, src_age_style = source_age_label(state, now_mono)
            line = Text("tape / --:--:-- ", style="dim")
            line.append("INFO", style="white")
            line.append(" no recent actions", style="dim")
            line.append(f" ({src_age_text})", style=src_age_style)
            line.append(" ", style="dim")
            line.append(f"[{runner_text}]", style=runner_style)
            tape_lines.append(line)
    else:
        for idx, event in enumerate(tape_events):
            ts_text = short_ts(event.ts_ms) if event.ts_ms else "--:--:--"
            tag_text, tag_style = _simple_tape_tag(event.kind)
            payload = _compact_tape_payload(event.text, tape_payload_width)
            line = Text(tape_prefix if idx == 0 else tape_indent, style="dim")
            line.append(f"{ts_text} ", style="white")

            flash_style = None
            if now_mono < event.flash_until_mono:
                flash_style = _flash_style(event.kind, pulse)

            if flash_style:
                line.append(tag_text, style=flash_style)
                line.append(" ")
                line.append(payload, style=flash_style)
            else:
                line.append(tag_text, style=tag_style)
                line.append(" ", style="dim")
                payload_style = "dim" if tag_text == "INFO" else "white"
                if event.kind in {"error", "kill"}:
                    payload_style = "bold red"
                line.append(payload, style=payload_style)
            tape_lines.append(line)

    delta_usd_value: float | None = None
    cap_value: float | None = None
    pos_base_value: float | None = None
    if not in_shadow:
        delta_usd_value = _extract_net_pos_usd(record)
        cap_value = _extract_max_pos_cap_usd(record)
        pos_base_value = _extract_net_pos_base(record)
    pos_text = _format_pos_short(pos_base_value) if not in_shadow else "n/a"
    delta_usd_text = _format_signed_dollars(delta_usd_value) if not in_shadow else "n/a"

    delta_pct: float | None = None
    if (
        not in_shadow
        and delta_usd_value is not None
        and cap_value is not None
        and cap_value > 0
    ):
        delta_pct = 100.0 * abs(delta_usd_value) / max(1.0, cap_value)
    delta_text = f"{delta_pct:.1f}%" if delta_pct is not None else "n/a"
    delta_style = _delta_pct_style(delta_pct) if not in_shadow else "dim"

    pos_style = "dim"
    if pos_base_value is not None and not in_shadow:
        if pos_base_value > 0:
            pos_style = "green"
        elif pos_base_value < 0:
            pos_style = "red"
        else:
            pos_style = "white"

    delta_usd_style = "dim"
    if delta_usd_value is not None and not in_shadow:
        if delta_usd_value > 0:
            delta_usd_style = "green"
        elif delta_usd_value < 0:
            delta_usd_style = "red"
        else:
            delta_usd_style = "white"

    pnl_label = "bPNL"
    pnl_text = _format_balance_pnl_short(state.balance_pnl)
    pnl_style = _balance_pnl_style(state.balance_pnl)
    if in_shadow:
        pnl_text = "n/a"
        pnl_style = "dim"
    elif state.balance_pnl.status not in {"available", "pending_post", "unreadable"}:
        pnl_label = "tPNL"
        pnl_value = _extract_pnl_usd(record)
        pnl_text = _format_signed_dollars(pnl_value)
        pnl_style = "dim"
        if pnl_value is not None:
            if pnl_value > 0:
                pnl_style = "green"
            elif pnl_value < 0:
                pnl_style = "red"
            else:
                pnl_style = "white"

    footer = Text("Δcap ")
    footer.append(delta_text, style=delta_style)
    footer.append(" // ")
    footer.append("Δ$", style="bold red")
    footer.append(" ")
    footer.append(delta_usd_text, style=delta_usd_style)
    footer.append(" // ")
    footer.append("POSb", style="bold red")
    footer.append(" ")
    footer.append(pos_text, style=pos_style)
    footer.append(" \\ ")
    footer.append(pnl_label, style="bold red")
    footer.append(" ")
    footer.append(pnl_text, style=pnl_style)
    footer.append(" // ")
    footer.append("sVOL", style="bold red")
    footer.append(" ")
    footer.append(str(state.run_fill_count), style="white")
    footer.append("f ", style="dim")
    footer.append(_format_base_volume_short(state.run_base_volume), style="white")
    footer.append("e ", style="dim")
    footer.append(_format_notional_short(state.run_notional_volume), style="white")
    aster_guard_text = _format_aster_guard_short(state)
    if aster_guard_text:
        guard_style = (
            "green"
            if state.aster_guard_decision == "allow"
            else "yellow"
            if state.aster_guard_decision == "suppress"
            else "dim"
        )
        footer.append(" // AG ", style="bold red")
        footer.append(aster_guard_text, style=guard_style)

    frame_lines: list[Text] = [header, legend, *tape_lines]
    rows_remaining = term_height - (len(frame_lines) + 1)
    quote_line = Text(QUOTE_FOOTER, style="dim italic")
    if rows_remaining >= 3:
        frame_lines.append(Text(""))
        frame_lines.append(quote_line)
        frame_lines.append(Text(""))
    frame_lines.append(footer)

    return Group(*frame_lines)


def render_frame_rich(
    state: WatchState,
    max_events: int,
    *,
    term_width: int,
    term_height: int,
    config_dir: str | None = None,
    health_url: str | None = None,
    layout_debug: bool = False,
    ui_mode: str | None = None,
    render_ms: int | None = None,
) -> Any:
    if state.active_page == _PAGE_EXPANDED:
        return render_frame_expanded(
            state,
            max_events,
            term_width=term_width,
            term_height=term_height,
            config_dir=config_dir,
            health_url=health_url,
            layout_debug=layout_debug,
            ui_mode=ui_mode,
            render_ms=render_ms,
        )
    return render_frame_simple(
        state,
        max_events,
        term_width=term_width,
        term_height=term_height,
        config_dir=config_dir,
        health_url=health_url,
        layout_debug=layout_debug,
        ui_mode=ui_mode,
        render_ms=render_ms,
    )


# ── Classic frame rendering ─────────────────────────────────────────────────


def render_frame_classic(  # noqa: C901
    state: WatchState,
    max_events: int,
    config_dir: str | None = None,
    health_url: str | None = None,
) -> str:
    state.frame_count += 1
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
    title = styled("  paraphina v1.1 watch", S.B_CYAN, S.BOLD)

    # ── Header metrics ────────────────────────────────────────────────────
    tick_str = styled(str(tick), S.B_WHITE) if tick is not None else styled("n/a", S.GRAY)
    time_str = (
        styled(short_ts(now_ms), S.B_WHITE, S.BOLD) if now_ms else styled("n/a", S.GRAY)
    )
    wall_time_str = styled(wall_clock_utc(), S.B_WHITE)
    src_age_text, src_age_style = source_age_label(state, time.monotonic())
    runner_text, runner_style = runner_status_label(state)
    src_age_str = styled(src_age_text, getattr(S, "YELLOW") if src_age_style == "yellow" else getattr(S, "B_RED") if src_age_style == "bold red" else getattr(S, "GREEN") if src_age_style == "green" else getattr(S, "WHITE") if src_age_style == "white" else S.DIM)
    runner_str = styled(runner_text, getattr(S, "YELLOW") if runner_style == "yellow" else getattr(S, "B_RED") if runner_style == "bold red" else getattr(S, "GREEN") if runner_style == "green" else getattr(S, "WHITE") if runner_style == "white" else S.DIM)
    mode_str = styled(str(execution_mode), S.MAGENTA)
    trade_str = styled(str(trade_mode), S.MAGENTA)
    regime_str = color_regime(str(risk_regime))
    kill_str = color_kill(bool(kill_switch))

    hdr1 = (
        f"  {_LABEL('tick')} {tick_str}   "
        f"{_LABEL('time')} {time_str}   "
        f"{src_age_str}   "
        f"{runner_str}   "
        f"{_LABEL('now')} {wall_time_str}   "
        f"{_LABEL('hb')} {styled(heartbeat_glyph(state.frame_count), S.CYAN)}   "
        f"{_LABEL('mode')} {mode_str}"
    )
    hdr2 = (
        f"  {_LABEL('trade')} {trade_str}   "
        f"{_LABEL('regime')} {regime_str}   {_LABEL('kill')} {kill_str}"
    )
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

    lines: list[str] = [rule, title, rule, hdr1, hdr2, hdr3, hdr4]

    # ── Deploy state (optional) ───────────────────────────────────────────
    if config_dir and health_url:
        deploy_lines = render_deploy_section(config_dir, health_url)
        lines.extend(deploy_lines)

    lines.append("")

    # ── Venue table ───────────────────────────────────────────────────────
    venue_ids = state.venue_ids
    v_status = record.get("venue_status", [])
    v_mid = record.get("venue_mid_usd", [])
    v_spread = record.get("venue_spread_usd", [])
    v_age = record.get("venue_age_ms", [])
    v_age_event = record.get("venue_age_event_ms", [])
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

    show_age_event = isinstance(v_age_event, list) and len(v_age_event) > 0

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
        age_event_val = v_age_event[idx] if idx < len(v_age_event) else None
        age_event_f = format_ms(age_event_val)
        age_event_f = _color_val(age_event_f, safe_float(safe_int(age_event_val)), 2000, 5000)
        pos_f = format_num(pos_val, 8).strip()
        pos_f = _color_val(pos_f, safe_float(pos_val), 0.1, 1.0)
        fund_rate_f = format_num(fund_rate_val, 8).strip()
        fund_age_f = format_ms(fund_age_val)
        fund_status_f = color_health(format_status(fund_status_val))
        orders_f = str(open_orders)

        row = [
            styled(venue_id, S.BOLD, S.WHITE),
            mid_f,
            spread_f,
            age_f,
        ]
        if show_age_event:
            row.append(age_event_f)
        row.extend(
            [
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
        rows.append(row)

    lines.append(_section("Venues"))
    headers = [
        "venue",
        "mid",
        "spread",
        "age_ms",
    ]
    if show_age_event:
        headers.append("age_event_ms")
    headers.extend(
        [
            "pos",
            "fund_8h",
            "fund_age",
            "fund_status",
            "orders",
            "last_fill",
            "health",
            "stale%/flips",
        ]
    )
    lines.append(format_table(headers, rows))

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


def _decode_key_stream(raw: str) -> tuple[list[str], str]:
    keys: list[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\x1b":
            if i + 1 >= len(raw):
                return keys, raw[i:]
            if raw[i + 1] == "[":
                if i + 2 >= len(raw):
                    return keys, raw[i:]
                code = raw[i + 2]
                if code == "D":
                    keys.append("LEFT")
                elif code == "C":
                    keys.append("RIGHT")
                i += 3
                continue
            i += 1
            continue
        keys.append(ch)
        i += 1
    return keys, ""


def _apply_watch_key(state: WatchState, key: str) -> None:
    low = key.lower()
    if low in {"1", "t"}:
        state.active_tab = _TAB_TAPE
    elif low in {"2", "k"}:
        state.active_tab = _TAB_KEYS
    elif low in {"3", "a"}:
        state.active_tab = _TAB_ALERTS
    elif key == "LEFT":
        state.active_tab = (state.active_tab - 1) % 3
    elif key == "RIGHT":
        state.active_tab = (state.active_tab + 1) % 3
    elif key == "?":
        state.show_vn_map_until = time.monotonic() + 8.0
    elif low == "x":
        state.active_page = _PAGE_SIMPLE
    elif low == "y":
        state.active_page = _PAGE_EXPANDED


class RichKeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self._fd: int | None = None
        self._old_term = None
        self._old_flags: int | None = None
        self._term_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buffer: Deque[str] = deque()
        self._lock = threading.Lock()
        self._pending = ""

    def start(self) -> None:
        if not sys.stdin.isatty() or termios is None or tty is None or fcntl is None:
            return
        self._stop.clear()
        try:
            fd = sys.stdin.fileno()
            old_term = termios.tcgetattr(fd)
            old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            with self._term_lock:
                self._fd = fd
                self._old_term = old_term
                self._old_flags = old_flags
            tty.setcbreak(fd)
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        except Exception:
            self.stop()
            return
        self.enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _restore_terminal(self) -> None:
        if termios is None or fcntl is None:
            return
        with self._term_lock:
            fd = self._fd
            old_term = self._old_term
            old_flags = self._old_flags
            self._fd = None
            self._old_term = None
            self._old_flags = None
        if fd is None:
            return
        try:
            if old_term is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            if old_flags is not None:
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
        except Exception:
            pass

    def _run(self) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            while not self._stop.is_set():
                try:
                    data = os.read(fd, 64)
                except BlockingIOError:
                    time.sleep(0.05)
                    continue
                except OSError:
                    break
                if not data:
                    time.sleep(0.05)
                    continue
                text = data.decode("utf-8", errors="ignore")
                with self._lock:
                    self._buffer.append(text)
        finally:
            self._stop.set()
            self._restore_terminal()

    def read_keys(self) -> list[str]:
        if not self.enabled:
            return []
        chunks: list[str] = []
        with self._lock:
            while self._buffer:
                chunks.append(self._buffer.popleft())
        raw = self._pending + "".join(chunks)
        keys, self._pending = _decode_key_stream(raw)
        return keys

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._restore_terminal()
        self.enabled = False


# ── Entry points ─────────────────────────────────────────────────────────────


def render_once(
    path: Path,
    refresh_ms: int,
    max_events: int,
    *,
    sort_mode: str = "agee",
    flash_ms: int = 650,
    use_rich: bool = False,
    config_dir: str | None = None,
    health_url: str | None = None,
    page: str = _PAGE_SIMPLE,
) -> str | Any:
    records = parse_lines(path, max_events)
    state = build_state(
        records,
        max_events,
        sort_mode=sort_mode,
        flash_ms=flash_ms,
        page=page,
    )
    seed_state_source_age_from_path(state, path)
    refresh_balance_pnl_from_run_dir(state, path.parent, force=True)
    if use_rich and _RICH_AVAILABLE:
        term_size = shutil.get_terminal_size((140, 40))
        return render_frame_rich(
            state,
            max_events,
            term_width=term_size.columns,
            term_height=term_size.lines,
            config_dir=config_dir,
            health_url=health_url,
        )
    return render_frame_classic(
        state,
        max_events,
        config_dir=config_dir,
        health_url=health_url,
    )


def main() -> int:
    global _NO_COLOR
    args = parse_args()
    try:
        telemetry_path = resolve_telemetry_path(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    max_events = max(1, args.max_events)
    refresh_ms = max(1, args.refresh_ms)
    flash_ms = max(150, args.flash_ms)
    use_rich = not args.classic
    auto_vscode_mode = detect_vscode_terminal()
    if args.vscode:
        vscode_mode = True
    elif args.no_vscode:
        vscode_mode = False
    else:
        vscode_mode = auto_vscode_mode
    render_ms = (
        max(1, args.render_ms)
        if args.render_ms is not None
        else (max(refresh_ms, 120) if vscode_mode else refresh_ms)
    )
    ui_mode = "vscode" if vscode_mode else None

    if args.no_color or os.environ.get("NO_COLOR"):
        _NO_COLOR = True

    if use_rich and not _RICH_AVAILABLE:
        print(
            "rich is not installed; falling back to classic renderer. "
            "Install with: pip install rich",
            file=sys.stderr,
        )
        use_rich = False

    # Deploy state panel (disabled with --no-deploy-state).
    deploy_config_dir: str | None = None
    deploy_health_url: str | None = None
    if not args.no_deploy_state:
        deploy_config_dir = args.config_dir
        deploy_health_url = args.health_url

    if not telemetry_path.exists():
        print(f"warning: telemetry path not found yet: {telemetry_path}", file=sys.stderr)

    one_shot = refresh_ms >= 999_999
    is_tty = sys.stdout.isatty()
    if not is_tty:
        _NO_COLOR = True

    initial_records = parse_lines(telemetry_path, max_events)
    state = build_state(
        initial_records,
        max_events,
        sort_mode=args.sort,
        flash_ms=flash_ms,
        page=args.page,
    )
    seed_state_source_age_from_path(state, telemetry_path)
    refresh_balance_pnl_from_run_dir(state, telemetry_path.parent, force=True)
    state.runner_status = load_runner_status(telemetry_path)
    follower = TailFollower(telemetry_path)
    # Advance follower past already-consumed data so we don't double-count.
    follower.seek_end()
    latest_check_interval_s = 1.0
    next_latest_check_ts = 0.0

    def _ingest_new_records() -> None:
        for line in follower.read_new_lines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                state.update(record)

    def _refresh_runner_status() -> None:
        state.runner_status = load_runner_status(telemetry_path)

    def _maybe_switch_auto_target(now_mono: float) -> None:
        nonlocal telemetry_path, follower, state, next_latest_check_ts
        if now_mono < next_latest_check_ts:
            return
        if args.current_run:
            resolver = resolve_current_run_telemetry_path
        elif args.latest:
            resolver = resolve_latest_telemetry_path
        else:
            return
        next_latest_check_ts = now_mono + latest_check_interval_s
        try:
            latest_path = resolver()
        except FileNotFoundError:
            if args.current_run:
                state.runner_status = RunnerStatus(
                    state="no_active",
                    alive=False,
                    status_path="current-run registry/process",
                )
            return
        if latest_path == telemetry_path:
            return
        state = rebuild_state_for_target(
            state,
            parse_lines(latest_path, max_events),
            max_events,
            flash_ms=flash_ms,
        )
        seed_state_source_age_from_path(state, latest_path)
        telemetry_path = latest_path
        refresh_balance_pnl_from_run_dir(state, telemetry_path.parent, force=True)
        follower = TailFollower(telemetry_path)
        follower.seek_end()
        _refresh_runner_status()

    if one_shot:
        if use_rich:
            console = Console(no_color=_NO_COLOR, force_terminal=is_tty, highlight=False)
            console.print(
                render_frame_rich(
                    state,
                    max_events,
                    term_width=console.size.width,
                    term_height=console.size.height,
                    config_dir=deploy_config_dir,
                    health_url=deploy_health_url,
                    layout_debug=args.layout_debug,
                    ui_mode=ui_mode,
                    render_ms=render_ms,
                )
            )
        else:
            print(
                render_frame_classic(
                    state,
                    max_events,
                    config_dir=deploy_config_dir,
                    health_url=deploy_health_url,
                )
            )
        return 0

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if use_rich:
        console = Console(no_color=_NO_COLOR, force_terminal=is_tty, highlight=False)
        key_reader = RichKeyReader()
        if is_tty:
            key_reader.start()
        refresh_interval_s = refresh_ms / 1000.0
        base_render_interval_s = render_ms / 1000.0
        drop_backoff_s = 0.015
        heartbeat_interval_s = 1.0
        vscode_min_render_interval_s = 0.150
        vscode_drop_cooldown_s = 2.0
        next_render_ts = 0.0
        next_heartbeat_ts = time.monotonic() + heartbeat_interval_s
        render_cooldown_until = 0.0
        last_rendered_key: tuple[Any, ...] | None = None

        def _effective_render_interval(now_mono: float) -> float:
            interval_s = base_render_interval_s
            if vscode_mode and now_mono < render_cooldown_until:
                interval_s = max(interval_s, vscode_min_render_interval_s)
            return interval_s

        try:
            if is_tty:
                for key in key_reader.read_keys():
                    _apply_watch_key(state, key)
                initial = render_frame_rich(
                    state,
                    max_events,
                    term_width=console.size.width,
                    term_height=console.size.height,
                    config_dir=deploy_config_dir,
                    health_url=deploy_health_url,
                    layout_debug=args.layout_debug,
                    ui_mode=ui_mode,
                    render_ms=render_ms,
                )
                with Live(
                    initial,
                    console=console,
                    auto_refresh=False,
                    transient=False,
                    screen=False,
                ) as live:
                    now = time.monotonic()
                    last_rendered_key = _build_frame_key(state, now)
                    next_render_ts = now + _effective_render_interval(now)
                    next_heartbeat_ts = now + heartbeat_interval_s
                    while True:
                        loop_started = time.monotonic()
                        _maybe_switch_auto_target(loop_started)
                        _ingest_new_records()
                        _refresh_runner_status()
                        refresh_balance_pnl_from_run_dir(state, telemetry_path.parent, now_mono=loop_started)
                        key_seen = False
                        for key in key_reader.read_keys():
                            _apply_watch_key(state, key)
                            key_seen = True
                        now = time.monotonic()
                        if key_seen:
                            next_render_ts = 0.0
                        if now >= next_render_ts:
                            frame_key = _build_frame_key(state, now)
                            force_render = key_seen or now >= next_heartbeat_ts
                            interval_s = _effective_render_interval(now)
                            if force_render or frame_key != last_rendered_key:
                                frame_dropped = False
                                try:
                                    live.update(
                                        render_frame_rich(
                                            state,
                                            max_events,
                                            term_width=console.size.width,
                                            term_height=console.size.height,
                                            config_dir=deploy_config_dir,
                                            health_url=deploy_health_url,
                                            layout_debug=args.layout_debug,
                                            ui_mode=ui_mode,
                                            render_ms=render_ms,
                                        ),
                                        refresh=False,
                                    )
                                    live.refresh()
                                except BlockingIOError:
                                    frame_dropped = True
                                except OSError as exc:
                                    if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                                        frame_dropped = True
                                    else:
                                        raise

                                now = time.monotonic()
                                if frame_dropped:
                                    if vscode_mode:
                                        render_cooldown_until = max(
                                            render_cooldown_until,
                                            now + vscode_drop_cooldown_s,
                                        )
                                    interval_s = _effective_render_interval(now)
                                    next_render_ts = now + max(drop_backoff_s, interval_s)
                                else:
                                    last_rendered_key = frame_key
                                    next_heartbeat_ts = now + heartbeat_interval_s
                                    next_render_ts = now + interval_s
                            else:
                                next_render_ts = now + interval_s

                        sleep_s = refresh_interval_s - (time.monotonic() - loop_started)
                        if sleep_s > 0:
                            time.sleep(sleep_s)
            else:
                while True:
                    loop_started = time.monotonic()
                    _maybe_switch_auto_target(loop_started)
                    _ingest_new_records()
                    _refresh_runner_status()
                    refresh_balance_pnl_from_run_dir(state, telemetry_path.parent, now_mono=loop_started)
                    now = time.monotonic()
                    if now >= next_render_ts:
                        frame_key = _build_frame_key(state, now)
                        force_render = now >= next_heartbeat_ts
                        interval_s = _effective_render_interval(now)
                        if force_render or frame_key != last_rendered_key:
                            term_size = shutil.get_terminal_size((140, 40))
                            frame_dropped = False
                            try:
                                console.print(
                                    render_frame_rich(
                                        state,
                                        max_events,
                                        term_width=term_size.columns,
                                        term_height=term_size.lines,
                                        config_dir=deploy_config_dir,
                                        health_url=deploy_health_url,
                                        layout_debug=args.layout_debug,
                                        ui_mode=ui_mode,
                                        render_ms=render_ms,
                                    )
                                )
                            except BlockingIOError:
                                frame_dropped = True
                            except OSError as exc:
                                if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                                    frame_dropped = True
                                else:
                                    raise

                            now = time.monotonic()
                            if frame_dropped:
                                if vscode_mode:
                                    render_cooldown_until = max(
                                        render_cooldown_until,
                                        now + vscode_drop_cooldown_s,
                                    )
                                interval_s = _effective_render_interval(now)
                                next_render_ts = now + max(drop_backoff_s, interval_s)
                            else:
                                last_rendered_key = frame_key
                                next_heartbeat_ts = now + heartbeat_interval_s
                                next_render_ts = now + interval_s
                        else:
                            next_render_ts = now + interval_s

                    sleep_s = refresh_interval_s - (time.monotonic() - loop_started)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        except KeyboardInterrupt:
            return 0
        finally:
            key_reader.stop()
        return 0

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
        _maybe_switch_auto_target(time.monotonic())
        loop_started = time.monotonic()
        for line in follower.read_new_lines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                state.update(record)
        _refresh_runner_status()
        refresh_balance_pnl_from_run_dir(state, telemetry_path.parent, now_mono=loop_started)
        frame = render_frame_classic(
            state,
            max_events,
            config_dir=deploy_config_dir, health_url=deploy_health_url,
        )
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
