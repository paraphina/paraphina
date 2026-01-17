#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMMANDS = [
    "cargo test --all",
    "cargo test -p paraphina --features live",
    "cargo test -p paraphina --features live,live_hyperliquid",
    "cargo test -p paraphina --features live,live_lighter",
    "cargo test -p paraphina --features live,live_paradex",
    "cargo test -p paraphina --features live,live_aster",
    "cargo test -p paraphina --features live,live_extended",
    "python3 -m pytest -q",
    "python3 tools/print_live_connector_matrix.py",
]


def read_step_b_results(path: Path | None) -> dict:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_connector_matrix() -> str:
    result = subprocess.run(
        [sys.executable, "tools/print_live_connector_matrix.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "connector_matrix_error"
    return result.stdout.strip()


def build_report(step_b_results: dict) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    connector_matrix = run_connector_matrix()
    feature_flags = [
        "live",
        "live_hyperliquid",
        "live_lighter",
        "live_paradex",
        "live_aster",
        "live_extended",
    ]
    exec_gates = [
        "preflight: PARAPHINA_LIVE_PREFLIGHT_OK=1",
        "enable flags: --enable-live-execution + PARAPHINA_LIVE_EXEC_ENABLE=1 + PARAPHINA_LIVE_EXECUTION_CONFIRM=YES",
        "canary profile: --canary-profile prod_canary (or PARAPHINA_LIVE_CANARY_PROFILE)",
        "reconciliation: PARAPHINA_LIVE_ACCOUNT_RECONCILE_MS > 0",
    ]
    how_to_run = [
        "PARAPHINA_TRADE_MODE=shadow PARAPHINA_LIVE_CONNECTOR=hyperliquid paraphina_live",
        "PARAPHINA_TRADE_MODE=paper PARAPHINA_LIVE_CONNECTOR=hyperliquid_fixture HL_FIXTURE_DIR=./tests/fixtures/hyperliquid paraphina_live",
        "cargo run -p paraphina --bin paraphina_live --features live,live_hyperliquid -- --preflight",
    ]
    last_results = step_b_results.get("results", [])
    if not last_results:
        last_results = [{"command": cmd, "status": "not_run"} for cmd in DEFAULT_COMMANDS]

    lines = [
        "# Roadmap‑B Readiness Report",
        "",
        f"- Generated (UTC): {timestamp}",
        "",
        "## Connector Matrix",
        "",
        connector_matrix or "connector_matrix_unavailable",
        "",
        "## Feature Flags",
        "",
        "- " + "\n- ".join(feature_flags),
        "",
        "## Execution Gates",
        "",
        "- " + "\n- ".join(exec_gates),
        "",
        "## How To Run",
        "",
        "- " + "\n- ".join(how_to_run),
        "",
        "## Last Known Test Results",
        "",
    ]
    for entry in last_results:
        cmd = entry.get("command", "unknown")
        status = entry.get("status", "unknown")
        detail = entry.get("detail", "")
        if detail:
            lines.append(f"- `{cmd}` → {status} ({detail})")
        else:
            lines.append(f"- `{cmd}` → {status}")
    lines.append("")
    if "step_b" in step_b_results:
        lines.append("## Step‑B Results")
        lines.append("")
        lines.append("```\n" + json.dumps(step_b_results.get("step_b", {}), indent=2) + "\n```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Roadmap-B readiness report (stdlib only)."
    )
    parser.add_argument(
        "--step-b-results",
        default="",
        help="Optional path to step_b_results.json",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs" / "ROADMAP_B_READINESS_REPORT.md"),
        help="Output report path",
    )
    args = parser.parse_args()

    path = Path(args.step_b_results) if args.step_b_results else None
    report = build_report(read_step_b_results(path))
    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
