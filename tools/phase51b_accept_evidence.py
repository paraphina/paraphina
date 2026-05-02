#!/usr/bin/env python3
"""Phase 5.1b evidence acceptance gate.

This tool promotes a read-only Lighter account/native-limit evidence pack only
to Phase 5.1c calibration-label ingestion. It never approves live, canary,
capital escalation, or risk-limit relaxation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EVENT_TYPES = {
    "V2_RUN_CONTEXT",
    "V2_LIGHTER_ACCOUNT_PROFILE",
    "V2_LIGHTER_ACCOUNT_LIMITS",
    "V2_LIGHTER_ACTIVE_ORDERS",
}
SENSITIVE_ENV_FRAGMENTS = ("PRIVATE", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object records in {path}")
            records.append(record)
    return records


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _schema_validation(telemetry_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_telemetry_contract.py"), str(telemetry_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _sensitive_env_values(env_path: Path) -> list[tuple[str, bytes]]:
    values: list[tuple[str, bytes]] = []
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not any(fragment in key.upper() for fragment in SENSITIVE_ENV_FRAGMENTS):
            continue
        try:
            parsed = shlex.split(raw_value, comments=False, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = raw_value.strip().strip("'\"")
        if len(value) >= 8:
            values.append((key, value.encode()))
    return values


def _secret_scan(run_dir: Path, env_path: Path | None) -> dict[str, Any]:
    if env_path is None:
        return {
            "performed": False,
            "sensitive_env_values_checked": 0,
            "sensitive_value_leak_found": None,
            "leak_locations_sanitized": [],
        }
    values = _sensitive_env_values(env_path)
    leaks: list[tuple[str, str]] = []
    for artifact in run_dir.rglob("*"):
        if not artifact.is_file():
            continue
        data = artifact.read_bytes()
        for key, value in values:
            if value in data:
                leaks.append((key, artifact.relative_to(run_dir).as_posix()))
    return {
        "performed": True,
        "sensitive_env_values_checked": len(values),
        "sensitive_value_leak_found": bool(leaks),
        "leak_locations_sanitized": sorted({f"{key}@{path}" for key, path in leaks}),
    }


def _all_safety_flags_hold(records: list[dict[str, Any]], gate: dict[str, Any]) -> bool:
    gate_safe = (
        gate.get("approved_for_live") is False
        and gate.get("approved_for_canary") is False
        and gate.get("approved_for_capital_escalation") is False
    )
    if not gate_safe:
        return False
    for record in records:
        if record.get("no_live_flag") is not True:
            return False
        if record.get("approved_for_live") is not False:
            return False
        if record.get("approved_for_canary") is not False:
            return False
        if record.get("approved_for_capital_escalation") is not False:
            return False
        if record.get("live_orders_allowed") is not False:
            return False
        if record.get("capital_change_allowed") is not False:
            return False
        if record.get("risk_limit_relaxation_allowed") is not False:
            return False
        if record.get("admissible_for_financial_claim") not in (None, False):
            return False
    return True


def _limitations(records: list[dict[str, Any]]) -> list[str]:
    by_type = {record.get("event_type"): record for record in records}
    limits = by_type.get("V2_LIGHTER_ACCOUNT_LIMITS", {})
    active = by_type.get("V2_LIGHTER_ACTIVE_ORDERS", {})
    limitations: list[str] = []
    if limits.get("sendtx_per_minute_limit") is None:
        limitations.append("lighter_sendtx_limit_not_exposed_by_account_limits_payload")
    if limits.get("rest_requests_per_minute_limit") is None:
        limitations.append("lighter_rest_request_limit_not_exposed_by_account_limits_payload")
    if active.get("open_order_limit_status") == "UNKNOWN":
        limitations.append("lighter_open_order_limit_headroom_unknown")
    return limitations


def accept(run_dir: Path, sensitive_env_file: Path | None, output: Path | None) -> Path:
    telemetry_path = run_dir / "telemetry.jsonl"
    gate_path = run_dir / "gate_result.json"
    manifest_path = run_dir / "manifest.json"
    if not telemetry_path.exists() or not gate_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("run directory must contain telemetry.jsonl, gate_result.json, and manifest.json")
    records = _load_jsonl(telemetry_path)
    gate = _load_json(gate_path)
    manifest = _load_json(manifest_path)
    event_types = {str(record.get("event_type")) for record in records}
    schema = _schema_validation(telemetry_path)
    secret_scan = _secret_scan(run_dir, sensitive_env_file)
    required_events_present = REQUIRED_EVENT_TYPES.issubset(event_types)
    capture_complete = gate.get("phase51b_capture_complete") is True
    safe_flags = _all_safety_flags_hold(records, gate)
    secret_scan_passed = secret_scan["performed"] and secret_scan["sensitive_value_leak_found"] is False
    accepted = all([
        schema["passed"],
        required_events_present,
        capture_complete,
        safe_flags,
        secret_scan_passed,
    ])
    result = {
        "schema_version": 1,
        "run_id": manifest.get("metadata", {}).get("run_id"),
        "run_dir": str(run_dir),
        "status": "PROMOTE_TO_PHASE51C_CALIBRATION_INGESTION" if accepted else "HOLD",
        "approved_for_calibration_label_ingestion": accepted,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_financial_claim": False,
        "required_event_types_present": required_events_present,
        "observed_event_types": sorted(event_types),
        "phase51b_capture_complete": capture_complete,
        "safe_nonlive_flags": safe_flags,
        "schema_validation": schema,
        "secret_scan": secret_scan,
        "limitations": _limitations(records),
        "verdict": (
            "Phase 5.1b evidence accepted only for Gate 5.1c calibration-label ingestion; "
            "live/canary/capital remain blocked."
            if accepted
            else "Phase 5.1b evidence remains HOLD."
        ),
    }
    out_path = output or (run_dir / "phase51b_acceptance.json")
    _write_json(out_path, result)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sensitive-env-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        out_path = accept(
            run_dir=args.run_dir,
            sensitive_env_file=args.sensitive_env_file,
            output=args.output,
        )
    except Exception as exc:
        print(f"phase51b_accept_evidence: ERROR: {exc}", file=sys.stderr)
        return 2
    result = _load_json(out_path)
    print(f"phase51b_accept_evidence: wrote {out_path}")
    print(f"phase51b_accept_evidence: status {result['status']}")
    return 0 if result["approved_for_calibration_label_ingestion"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
