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
    "cargo test -p paraphina --features live,live_hyperliquid,live_lighter,live_aster,live_extended,live_paradex",
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


def load_default_results() -> dict:
    default_path = ROOT / "agent-tools" / "step_b_results.json"
    if default_path.exists():
        return read_step_b_results(default_path)
    return {}


def normalize_results(raw: dict, commands: list[str]) -> list[dict]:
    results = raw.get("results", [])
    if not results:
        return [{"command": cmd, "status": "not_run"} for cmd in commands]
    lookup = {}
    for entry in results:
        cmd = entry.get("command")
        if not cmd:
            continue
        exit_code = entry.get("exit_code")
        status = "passed" if exit_code == 0 else "failed"
        lookup[cmd] = {
            "command": cmd,
            "status": status,
            "detail": entry.get("summary", ""),
        }
    return [lookup.get(cmd, {"command": cmd, "status": "not_run"}) for cmd in commands]


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


def parse_markdown_table(table: str) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in table.splitlines() if "|" in line]
    if len(lines) < 2:
        return []
    header_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < len(header_cells):
            continue
        if len(cells) > len(header_cells):
            head = cells[: len(header_cells) - 1]
            tail = " | ".join(cells[len(header_cells) - 1 :])
            cells = head + [tail]
        rows.append(dict(zip(header_cells, cells)))
    return rows


def readiness_stage(row: dict[str, str]) -> str:
    support = row.get("Support", "")
    trade_modes = row.get("Trade modes", "")
    execution = row.get("Execution", "")
    cancel_all = row.get("Cancel all", "")
    notes = row.get("Notes", "").lower()
    connector = row.get("Connector", "").lower()
    if support == "MissingFeature":
        return "MissingFeature"
    if support == "Stub":
        return "Stub"
    if "fixture" in connector or "fixture-only" in notes or "fixture-driven" in notes or "offline replay only" in notes:
        return "FixtureMarket+Account"
    if execution.startswith("No") and "Yes" in row.get("Market data", ""):
        return "ShadowLiveMarket"
    if execution.startswith("Yes") and "Live" not in trade_modes:
        return "PaperExec"
    if execution.startswith("Yes") and "Live" in trade_modes:
        if cancel_all.startswith("Yes"):
            return "CanaryLive"
        return "PaperExec"
    if support == "MarketOnly":
        return "ShadowLiveMarket"
    if support == "Market+Account":
        return "FixtureMarket+Account"
    return "Stub"


def connector_blockers(row: dict[str, str]) -> list[str]:
    blockers: list[str] = []
    support = row.get("Support", "")
    trade_modes = row.get("Trade modes", "")
    execution = row.get("Execution", "")
    cancel_all = row.get("Cancel all", "")
    notes = row.get("Notes", "").lower()
    connector = row.get("Connector", "").lower()
    if support == "MissingFeature":
        blockers.append("feature flag missing")
        return blockers
    if support == "Stub":
        blockers.append("connector stub")
        return blockers
    if "fixture" in connector or "fixture-only" in notes or "fixture-driven" in notes or "offline replay only" in notes:
        blockers.append("fixture-only (no live endpoints)")
    if execution.startswith("No"):
        blockers.append("execution not wired")
    if "Live" not in trade_modes:
        blockers.append("live trade mode unavailable")
    if cancel_all.startswith("No"):
        blockers.append("cancel_all unsupported")
    if support == "MarketOnly":
        blockers.append("account ingestion missing")
    if execution.startswith("Yes") and "Live" in trade_modes and cancel_all.startswith("Yes"):
        blockers.append("scaled live validation pending")
    return blockers


def build_report(step_b_results: dict) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    connector_matrix = run_connector_matrix()
    matrix_rows = parse_markdown_table(connector_matrix)
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
    if not step_b_results:
        step_b_results = load_default_results()
    last_results = normalize_results(step_b_results, DEFAULT_COMMANDS)
    results_by_command = {entry.get("command"): entry.get("status") for entry in last_results}

    lines = [
        "# Roadmap‑B Readiness Report",
        "",
        f"- Generated (UTC): {timestamp}",
        "",
        "## Connector Matrix",
        "",
        connector_matrix or "connector_matrix_unavailable",
        "",
        "## Feature Gap & Connector Readiness Ladder",
        "",
        "Ladder:",
        "",
        "- MissingFeature → Stub → FixtureMarket+Account → ShadowLiveMarket → PaperExec → CanaryLive → ScaledLive",
        "",
        "Connector stages:",
    ]
    if matrix_rows:
        for row in matrix_rows:
            connector = row.get("Connector", "unknown")
            stage = readiness_stage(row)
            lines.append(f"- {connector}: {stage}")
    else:
        lines.append("- connector_matrix_unavailable")
    lines.extend([
        "",
        "## All-5 Readiness Gates",
        "",
    ])
    connectors = ["hyperliquid", "lighter", "extended", "aster", "paradex"]
    stage_lookup = {
        row.get("Connector", "").lower(): readiness_stage(row) for row in matrix_rows
    }
    stage_ok = all(
        stage_lookup.get(conn, "") in {"PaperExec", "CanaryLive", "ScaledLive"}
        for conn in connectors
    )
    all5_features_cmd = "cargo test -p paraphina --features live,live_hyperliquid,live_lighter,live_aster,live_extended,live_paradex"
    all5_status = results_by_command.get(all5_features_cmd, "not_run")
    all5_paperexec = "PASS" if stage_ok and all5_status == "passed" else "FAIL"
    if all5_status == "not_run":
        all5_paperexec = "UNKNOWN"
    per_connector_cmds = [
        "cargo test -p paraphina --features live,live_hyperliquid",
        "cargo test -p paraphina --features live,live_lighter",
        "cargo test -p paraphina --features live,live_paradex",
        "cargo test -p paraphina --features live,live_aster",
        "cargo test -p paraphina --features live,live_extended",
    ]
    per_connector_status = [results_by_command.get(cmd, "not_run") for cmd in per_connector_cmds]
    all5_compile = "PASS" if per_connector_status and all(s == "passed" for s in per_connector_status) else "FAIL"
    if any(s == "not_run" for s in per_connector_status):
        all5_compile = "PASS" if all5_status == "passed" else "UNKNOWN"
    canary_ready = [conn for conn in connectors if stage_lookup.get(conn, "") == "CanaryLive"]
    lines.extend([
        f"- All-5 compile under feature flags: {all5_compile}",
        f"- All-5 PaperExec (offline deterministic): {all5_paperexec}",
        "- CanaryLive-ready connectors: " + (", ".join(canary_ready) if canary_ready else "none"),
        "",
    ])
    lines.extend([
        "",
        "## Connector Blockers",
        "",
    ])
    if matrix_rows:
        for row in matrix_rows:
            connector = row.get("Connector", "unknown")
            blockers = connector_blockers(row)
            if blockers:
                lines.append(f"- {connector}: " + "; ".join(blockers))
            else:
                lines.append(f"- {connector}: none")
    else:
        lines.append("- connector_matrix_unavailable")
    lines.extend([
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
    ])
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
