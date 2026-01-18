#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def load_template() -> str:
    template_path = ROOT / "docs" / "BURNIN_REPORT_TEMPLATE.md"
    return template_path.read_text(encoding="utf-8")


def count_telemetry_stats(path: Path) -> tuple[int, int, int, int, dict[str, dict[str, int]]]:
    if not path.exists():
        return 0, 0, 0, 0, {}
    fills = 0
    drift = 0
    cancels = 0
    rows = 0
    rollups: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        data = json.loads(line)
        for fill in data.get("fills", []):
            fills += 1
            venue_id = fill.get("venue_id", "unknown")
            venue = rollups.setdefault(venue_id, {"fills": 0, "cancels": 0, "drift": 0})
            venue["fills"] += 1
        for order in data.get("orders", []):
            if order.get("action") == "cancel" and order.get("status") == "ack":
                cancels += 1
                venue_id = order.get("venue_id", "unknown")
                venue = rollups.setdefault(venue_id, {"fills": 0, "cancels": 0, "drift": 0})
                venue["cancels"] += 1
        for drift_event in data.get("reconcile_drift", []):
            drift += 1
            venue_id = drift_event.get("venue_id", "unknown")
            venue = rollups.setdefault(venue_id, {"fills": 0, "cancels": 0, "drift": 0})
            venue["drift"] += 1
    return rows, fills, drift, cancels, rollups


def render_venue_rollups(rollups: dict[str, dict[str, int]]) -> str:
    if not rollups:
        return "n/a"
    lines = ["| Venue | Fills | Cancels | Drift |", "| --- | --- | --- | --- |"]
    for venue in sorted(rollups.keys()):
        stats = rollups[venue]
        lines.append(
            f"| {venue} | {stats.get('fills', 0)} | {stats.get('cancels', 0)} | {stats.get('drift', 0)} |"
        )
    return "\n".join(lines)


def read_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run burn-in harness and generate report.")
    parser.add_argument("--mode", choices=["shadow", "paper"], required=True)
    parser.add_argument("--connector", default="")
    parser.add_argument("--connectors", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ticks", type=int, default=25)
    parser.add_argument("--features", default="live")
    parser.add_argument("--fixture-dir", default="")
    parser.add_argument("--hl-fixture-dir", default="")
    parser.add_argument("--lighter-fixture-dir", default="")
    parser.add_argument("--roadmap-b-fixture-dir", default="")
    parser.add_argument("--paper-fill-mode", default="")
    parser.add_argument("--paper-slippage-bps", default="")
    parser.add_argument("--paper-smoke-intents", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry.jsonl"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "burnin_report.md"

    env = os.environ.copy()
    env["PARAPHINA_LIVE_OUT_DIR"] = str(out_dir)
    env["PARAPHINA_TELEMETRY_MODE"] = "jsonl"
    env["PARAPHINA_LIVE_MAX_TICKS"] = str(args.ticks)
    if args.fixture_dir and not args.hl_fixture_dir:
        env["HL_FIXTURE_DIR"] = args.fixture_dir
    if args.hl_fixture_dir:
        env["HL_FIXTURE_DIR"] = args.hl_fixture_dir
    if args.lighter_fixture_dir:
        env["LIGHTER_FIXTURE_DIR"] = args.lighter_fixture_dir
    if args.roadmap_b_fixture_dir:
        env["ROADMAP_B_FIXTURE_DIR"] = args.roadmap_b_fixture_dir
    if args.mode == "paper":
        if args.paper_fill_mode:
            env["PARAPHINA_PAPER_FILL_MODE"] = args.paper_fill_mode
        if args.paper_slippage_bps:
            env["PARAPHINA_PAPER_SLIPPAGE_BPS"] = args.paper_slippage_bps
        if args.paper_smoke_intents:
            env["PARAPHINA_PAPER_SMOKE_INTENTS"] = "1"

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    if args.connectors:
        connector_label = args.connectors
        env["PARAPHINA_LIVE_CONNECTORS"] = args.connectors
        connectors = [c.strip() for c in args.connectors.split(",") if c.strip()]
        if "extended" in connectors:
            env["EXTENDED_FIXTURE_MODE"] = "1"
        if "aster" in connectors:
            env["ASTER_FIXTURE_MODE"] = "1"
        if "paradex" in connectors:
            env["PARADEX_FIXTURE_MODE"] = "1"
    else:
        if not args.connector:
            raise SystemExit("Expected --connector or --connectors")
        connector_label = args.connector

    live_cmd = [
        "cargo",
        "run",
        "-p",
        "paraphina",
        "--bin",
        "paraphina_live",
        "--features",
        ",".join(features),
        "--",
        "--trade-mode",
        args.mode,
    ]
    if not args.connectors:
        live_cmd.extend(["--connector", args.connector])

    run = run_cmd(live_cmd, env)
    telemetry_check = run_cmd(
        [sys.executable, "tools/check_telemetry_contract.py", str(telemetry_path)],
        env,
    )

    summary = read_summary(summary_path)
    rows, fills, drift, cancels, rollups = count_telemetry_stats(telemetry_path)
    venue_rollups = render_venue_rollups(rollups)
    execution_mode = summary.get("execution_mode", "")
    ticks_run = summary.get("ticks_run", 0)
    kill_switch = summary.get("kill_switch", False)
    fv_available_rate = summary.get("fv_available_rate", 0.0)

    gate_telemetry = telemetry_check.returncode == 0
    gate_kill = not bool(kill_switch)
    gate_execution = execution_mode == args.mode if execution_mode else False

    report = load_template().format(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode=args.mode,
        connector=connector_label,
        features=",".join(features),
        out_dir=str(out_dir),
        ticks=args.ticks,
        fixture_dir=args.fixture_dir or "n/a",
        run_exit_code=run.returncode,
        telemetry_contract="PASS" if gate_telemetry else "FAIL",
        summary_present="yes" if summary else "no",
        ticks_run=ticks_run,
        kill_switch=kill_switch,
        execution_mode=execution_mode or "unknown",
        fv_available_rate=fv_available_rate,
        fills_count=fills,
        drift_count=drift,
        cancel_count=cancels,
        venue_rollups=venue_rollups,
        gate_telemetry="PASS" if gate_telemetry else "FAIL",
        gate_kill_switch="PASS" if gate_kill else "FAIL",
        gate_execution_mode="PASS" if gate_execution else "FAIL",
        notes=f"telemetry_rows={rows}",
    )
    report_path.write_text(report, encoding="utf-8")

    if run.returncode != 0 or not gate_telemetry or not gate_kill or not gate_execution:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
