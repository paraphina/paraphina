#!/usr/bin/env python3
"""Offline terminal-flatness gate for V2 live canary closeout.

The gate consumes a saved, sanitized direct venue-audit JSON artifact. It does
not call venue APIs, load credentials, infer fills, or mutate orders. Its job is
to prevent a V2 authority-pass artifact from being treated as promotion-ready
when terminal venue truth is missing, stale, non-flat, or has open orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


class TerminalFlatnessGateError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TerminalFlatnessGateError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TerminalFlatnessGateError("venue audit artifact must be a JSON object")
    return data


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(float(parsed)) else None


def _manifest_run_binding_reasons(
    *,
    canary_manifest_path: Path | None,
    venue_audit_path: Path,
    run_token: str | None,
    audit_captured_after_run: bool,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    manifest_meta: dict[str, Any] = {
        "canary_manifest_path": str(canary_manifest_path) if canary_manifest_path else None,
        "canary_manifest_sha256": None,
        "run_token": run_token,
        "audit_captured_after_run_confirmed": audit_captured_after_run,
    }
    token = run_token.strip() if isinstance(run_token, str) else ""
    if not token:
        reasons.append("run_binding_missing_token")
    if canary_manifest_path is None:
        reasons.append("run_binding_missing_canary_manifest")
    elif not canary_manifest_path.exists():
        reasons.append("run_binding_canary_manifest_missing")
    else:
        manifest_meta["canary_manifest_sha256"] = _sha256(canary_manifest_path)
        try:
            manifest = _load_json(canary_manifest_path)
        except TerminalFlatnessGateError:
            reasons.append("run_binding_canary_manifest_invalid")
            manifest = {}
        if manifest.get("artifact_type") != "v2_authority_decision_evidence_manifest":
            reasons.append("run_binding_canary_manifest_wrong_type")
        if manifest.get("decision_validation_status") != "pass":
            reasons.append("run_binding_canary_manifest_not_pass")
        governance = manifest.get("governance")
        if not isinstance(governance, dict):
            reasons.append("run_binding_canary_manifest_missing_governance")
        else:
            if governance.get("approved_for_promotion") is True:
                reasons.append("run_binding_canary_manifest_already_promotion")
            if governance.get("blocker_cleared") is True:
                reasons.append("run_binding_canary_manifest_blocker_cleared")
            if governance.get("pressure_complete_claim") is True:
                reasons.append("run_binding_canary_manifest_pressure_complete")
        if token:
            candidate_text = str(canary_manifest_path)
            files = manifest.get("files")
            if isinstance(files, list):
                for item in files:
                    if isinstance(item, dict):
                        candidate_text += "\n" + str(item.get("path", ""))
            if token not in candidate_text:
                reasons.append("run_binding_token_not_in_canary_manifest")
    if token and token not in str(venue_audit_path):
        reasons.append("run_binding_token_not_in_venue_audit_path")
    if not audit_captured_after_run:
        reasons.append("run_binding_after_run_confirmation_missing")
    return reasons, manifest_meta


def _venue_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = data.get("results")
    if not isinstance(results, list):
        raise TerminalFlatnessGateError("venue audit artifact missing results[]")
    mapped: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise TerminalFlatnessGateError("venue audit results[] entries must be objects")
        venue = item.get("venue")
        if not isinstance(venue, str) or not venue.strip():
            raise TerminalFlatnessGateError("venue audit result missing venue")
        normalized = venue.strip().lower()
        if normalized in mapped:
            raise TerminalFlatnessGateError(f"duplicate venue result: {normalized}")
        mapped[normalized] = item
    return mapped


def evaluate_terminal_flatness(
    *,
    venue_audit_path: Path,
    expected_venues: list[str],
    position_tol_base: float,
    canary_manifest_path: Path | None = None,
    run_token: str | None = None,
    audit_captured_after_run: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(position_tol_base) or position_tol_base < 0:
        raise TerminalFlatnessGateError("position tolerance must be finite and non-negative")
    data = _load_json(venue_audit_path)
    expected = [venue.strip().lower() for venue in expected_venues if venue.strip()]
    if not expected:
        raise TerminalFlatnessGateError("at least one expected venue is required")
    observed = _venue_map(data)
    reasons: list[str] = []
    venue_rows: list[dict[str, Any]] = []
    binding_reasons, binding_meta = _manifest_run_binding_reasons(
        canary_manifest_path=canary_manifest_path,
        venue_audit_path=venue_audit_path,
        run_token=run_token,
        audit_captured_after_run=audit_captured_after_run,
    )
    reasons.extend(binding_reasons)

    if data.get("ok") is not True:
        reasons.append("venue_audit_not_ok")

    audit_tol = _as_float(data.get("position_tol_base"))
    if audit_tol is None:
        reasons.append("venue_audit_missing_position_tolerance")
    elif abs(audit_tol - position_tol_base) > 1e-12:
        reasons.append("venue_audit_position_tolerance_mismatch")

    for venue in expected:
        result = observed.get(venue)
        if result is None:
            reasons.append(f"{venue}:missing_venue_result")
            venue_rows.append(
                {
                    "venue": venue,
                    "gate_status": "HOLD",
                    "reason": "missing_venue_result",
                }
            )
            continue

        row_reasons: list[str] = []
        position = _as_float(result.get("position_base"))
        open_orders = _as_int(result.get("open_order_count"))
        open_orders_known = result.get("open_order_count_known") is True
        result_ok = result.get("ok") is True

        if not result_ok:
            row_reasons.append("venue_result_not_ok")
        if position is None:
            row_reasons.append("position_base_missing")
        elif abs(position) > position_tol_base + 1e-12:
            row_reasons.append("position_base_not_flat")
        if not open_orders_known:
            row_reasons.append("open_order_count_unknown")
        if open_orders is None:
            row_reasons.append("open_order_count_missing")
        elif open_orders != 0:
            row_reasons.append("open_orders_present")

        if row_reasons:
            reasons.extend(f"{venue}:{reason}" for reason in row_reasons)
        venue_rows.append(
            {
                "venue": venue,
                "market": result.get("market"),
                "position_abs": abs(position) if position is not None else None,
                "open_order_count": open_orders,
                "open_order_count_known": open_orders_known,
                "gate_status": "PASS" if not row_reasons else "HOLD",
                "reasons": row_reasons,
            }
        )

    status = "PASS" if not reasons else "HOLD"
    return {
        "artifact_type": "v2_live_canary_terminal_flatness_gate",
        "schema_version": 1,
        "terminal_flatness_gate_status": status,
        "terminal_flatness_gate_reasons": reasons,
        "expected_venues": expected,
        "position_tol_base": position_tol_base,
        "venue_results": venue_rows,
        "inputs": {
            "venue_audit_path": str(venue_audit_path),
            "venue_audit_sha256": _sha256(venue_audit_path),
            **binding_meta,
        },
        "governance": {
            "approved_for_promotion": False,
            "approved_for_live": False,
            "approved_for_capital_escalation": False,
            "blocker_cleared": False,
            "pressure_complete_claim": False,
            "v2_authority_admission": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "terminal_flatness_is_operational_closeout_only": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue-audit", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument(
        "--audit-captured-after-run",
        action="store_true",
        help="Required explicit confirmation that the venue audit was captured after canary exit/cleanup.",
    )
    parser.add_argument("--expected-venues", required=True)
    parser.add_argument("--position-tol-base", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = evaluate_terminal_flatness(
            venue_audit_path=args.venue_audit,
            expected_venues=args.expected_venues.split(","),
            position_tol_base=args.position_tol_base,
            canary_manifest_path=args.canary_manifest,
            run_token=args.run_token,
            audit_captured_after_run=args.audit_captured_after_run,
        )
    except TerminalFlatnessGateError as exc:
        print(f"V2_TERMINAL_FLATNESS_GATE_ERROR: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["terminal_flatness_gate_status"] != "PASS":
        print(
            "V2_TERMINAL_FLATNESS_GATE_HOLD "
            f"reasons={','.join(report['terminal_flatness_gate_reasons'])}"
        )
        return 1
    print("V2_TERMINAL_FLATNESS_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
