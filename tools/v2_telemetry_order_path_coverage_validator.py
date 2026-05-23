#!/usr/bin/env python3
"""Validate V2 telemetry-only live venue order-path coverage.

This validator is intentionally separate from V2 authority-decision validation.
It consumes saved telemetry, preflight text, summary JSON, and a saved direct
venue-audit artifact. It does not call venue APIs, load credentials, infer
fills, infer authority decisions, or approve promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SENSITIVE_KEY_MARKERS = (
    "client_order_id",
    "venue_order_id",
    "order_id",
    "trade_id",
    "fill_id",
    "private_key",
    "signature",
    "auth_token",
    "secret",
    "raw_payload",
    "raw_response",
    "headers",
)

PREFLIGHT_REQUIRED_SNIPPETS = (
    "- PASS v2_live_canary_admission",
    "venue_coverage_replacements_disabled=true",
    "venue_coverage_probe_approved=true",
    "venue_coverage_probe_venues_present=true",
    "ranked_execution_venues_present=true",
    "exit_cancel_all=true",
    "exit_position_flatten=true",
)


class V2TelemetryCoverageError(ValueError):
    pass


@dataclass
class CoverageCounts:
    telemetry_rows: int = 0
    target_place_intents: int = 0
    target_place_acks: int = 0
    target_cancel_intents: int = 0
    target_cancel_acks: int = 0
    target_replace_events: int = 0
    target_fills: int = 0
    target_would_send_places: int = 0
    target_would_send_cancels: int = 0
    max_place_size: float | None = None
    place_intent_sides: set[str] = field(default_factory=set)
    observed_order_statuses: set[str] = field(default_factory=set)
    observed_order_actions: set[str] = field(default_factory=set)
    bad_place_policy_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "telemetry_rows": self.telemetry_rows,
            "target_place_intents": self.target_place_intents,
            "target_place_acks": self.target_place_acks,
            "target_cancel_intents": self.target_cancel_intents,
            "target_cancel_acks": self.target_cancel_acks,
            "target_replace_events": self.target_replace_events,
            "target_fills": self.target_fills,
            "target_would_send_places": self.target_would_send_places,
            "target_would_send_cancels": self.target_would_send_cancels,
            "max_place_size": self.max_place_size,
            "place_intent_sides": sorted(self.place_intent_sides),
            "observed_order_statuses": sorted(self.observed_order_statuses),
            "observed_order_actions": sorted(self.observed_order_actions),
            "bad_place_policy_count": self.bad_place_policy_count,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V2TelemetryCoverageError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise V2TelemetryCoverageError(f"{path}: expected JSON object")
    return value


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


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in SENSITIVE_KEY_MARKERS):
                return True
            if _contains_sensitive_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _check_preflight(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        f"preflight_missing:{snippet}"
        for snippet in PREFLIGHT_REQUIRED_SNIPPETS
        if snippet not in text
    ]


def _load_telemetry_counts(path: Path, target_venue: str, max_order_size: float | None) -> tuple[CoverageCounts, list[str]]:
    counts = CoverageCounts()
    reasons: list[str] = []
    target = target_venue.strip().lower()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise V2TelemetryCoverageError(f"{path}: cannot open telemetry: {exc}") from exc
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V2TelemetryCoverageError(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise V2TelemetryCoverageError(f"{path}: line {line_no}: expected JSON object")
            counts.telemetry_rows += 1
            if _contains_sensitive_key(row):
                reasons.append(f"telemetry_sensitive_key_present_line:{line_no}")
                continue
            for fill in row.get("fills") or []:
                if isinstance(fill, dict) and str(fill.get("venue_id", "")).lower() == target:
                    counts.target_fills += 1
            for order in row.get("orders") or []:
                if not isinstance(order, dict):
                    continue
                if str(order.get("venue_id", "")).lower() != target:
                    continue
                action = str(order.get("action", "")).lower()
                status = str(order.get("status", "")).lower()
                counts.observed_order_actions.add(action)
                counts.observed_order_statuses.add(status)
                if action == "replace":
                    counts.target_replace_events += 1
                if action == "place" and status == "intent":
                    counts.target_place_intents += 1
                    side = order.get("side")
                    if isinstance(side, str) and side:
                        counts.place_intent_sides.add(side)
                    size = _as_float(order.get("size"))
                    if size is not None:
                        counts.max_place_size = max(size, counts.max_place_size or size)
                    if (
                        order.get("post_only") is not True
                        or order.get("reduce_only") is not False
                        or str(order.get("tif", "")).lower() != "gtc"
                    ):
                        counts.bad_place_policy_count += 1
                    if max_order_size is not None and size is not None and size > max_order_size + 1e-12:
                        reasons.append("target_place_size_exceeds_limit")
                elif action == "place" and status == "ack":
                    counts.target_place_acks += 1
                elif action == "cancel" and status == "intent":
                    counts.target_cancel_intents += 1
                elif action == "cancel" and status == "ack":
                    counts.target_cancel_acks += 1
            for order in row.get("would_send_orders") or []:
                if not isinstance(order, dict):
                    continue
                if str(order.get("venue_id", "")).lower() != target:
                    continue
                action = str(order.get("action", "")).lower()
                if action == "place":
                    counts.target_would_send_places += 1
                elif action == "cancel":
                    counts.target_would_send_cancels += 1
                elif action == "replace":
                    counts.target_replace_events += 1
    return counts, reasons


def _check_summary(path: Path) -> tuple[dict[str, Any], list[str]]:
    summary = _load_json(path)
    reasons: list[str] = []
    if _contains_sensitive_key(summary):
        reasons.append("summary_sensitive_key_present")
    if _as_int(summary.get("ticks_run")) is None:
        reasons.append("summary_missing_ticks_run")
    if _as_int(summary.get("kill_events")) not in (0,):
        reasons.append("summary_kill_events_present")
    replace_by_purpose = summary.get("would_replace_by_purpose")
    if replace_by_purpose not in ({}, None):
        reasons.append("summary_replace_intents_present")
    if str(summary.get("execution_mode", "")).lower() != "live":
        reasons.append("summary_execution_mode_not_live")
    return summary, reasons


def _check_venue_audit(path: Path, expected_venues: list[str], position_tol_base: float) -> tuple[list[dict[str, Any]], list[str]]:
    audit = _load_json(path)
    reasons: list[str] = []
    if _contains_sensitive_key(audit):
        reasons.append("venue_audit_sensitive_key_present")
    if audit.get("ok") is not True:
        reasons.append("venue_audit_not_ok")
    audit_tol = _as_float(audit.get("position_tol_base"))
    if audit_tol is None or abs(audit_tol - position_tol_base) > 1e-12:
        reasons.append("venue_audit_position_tolerance_mismatch")
    results = audit.get("results")
    if not isinstance(results, list):
        raise V2TelemetryCoverageError("venue audit missing results[]")
    by_venue = {
        str(item.get("venue", "")).lower(): item
        for item in results
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for venue in expected_venues:
        normalized = venue.strip().lower()
        result = by_venue.get(normalized)
        if result is None:
            reasons.append(f"{normalized}:missing_venue_result")
            rows.append({"venue": normalized, "gate_status": "HOLD", "reasons": ["missing_venue_result"]})
            continue
        row_reasons: list[str] = []
        position = _as_float(result.get("position_base"))
        open_orders = _as_int(result.get("open_order_count"))
        if result.get("ok") is not True:
            row_reasons.append("venue_result_not_ok")
        if position is None or abs(position) > position_tol_base + 1e-12:
            row_reasons.append("position_base_not_flat")
        if result.get("open_order_count_known") is not True:
            row_reasons.append("open_order_count_unknown")
        if open_orders is None or open_orders != 0:
            row_reasons.append("open_orders_present")
        if row_reasons:
            reasons.extend(f"{normalized}:{reason}" for reason in row_reasons)
        rows.append({
            "venue": normalized,
            "market": result.get("market"),
            "position_abs": abs(position) if position is not None else None,
            "open_order_count": open_orders,
            "gate_status": "PASS" if not row_reasons else "HOLD",
            "reasons": row_reasons,
        })
    return rows, reasons


def validate_order_path_coverage(
    *,
    telemetry_path: Path,
    preflight_path: Path,
    summary_path: Path,
    venue_audit_path: Path,
    target_venue: str,
    expected_venues: list[str],
    position_tol_base: float,
    max_order_size: float | None,
    run_token: str,
    audit_captured_after_run: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    token = run_token.strip()
    if not token:
        reasons.append("run_binding_missing_token")
    else:
        for label, path in (
            ("telemetry", telemetry_path),
            ("preflight", preflight_path),
            ("summary", summary_path),
            ("venue_audit", venue_audit_path),
        ):
            if token not in str(path):
                reasons.append(f"run_binding_token_not_in_{label}_path")
    if not audit_captured_after_run:
        reasons.append("run_binding_after_run_confirmation_missing")

    reasons.extend(_check_preflight(preflight_path))
    summary, summary_reasons = _check_summary(summary_path)
    reasons.extend(summary_reasons)
    counts, telemetry_reasons = _load_telemetry_counts(telemetry_path, target_venue, max_order_size)
    reasons.extend(telemetry_reasons)
    venue_rows, venue_reasons = _check_venue_audit(venue_audit_path, expected_venues, position_tol_base)
    reasons.extend(venue_reasons)

    if counts.telemetry_rows <= 0:
        reasons.append("telemetry_empty")
    if counts.target_place_intents <= 0:
        reasons.append("target_place_intent_missing")
    if counts.target_place_acks <= 0:
        reasons.append("target_place_ack_missing")
    if counts.target_cancel_intents <= 0:
        reasons.append("target_cancel_intent_missing")
    if counts.target_cancel_acks <= 0:
        reasons.append("target_cancel_ack_missing")
    if counts.target_replace_events:
        reasons.append("target_replace_events_present")
    if counts.bad_place_policy_count:
        reasons.append("target_place_policy_not_strict_post_only_gtc")

    status = "PASS" if not reasons else "HOLD"
    return {
        "artifact_type": "v2_telemetry_order_path_coverage_manifest",
        "schema_version": 1,
        "validation_status": status,
        "validation_reasons": reasons,
        "scope": {
            "target_venue": target_venue.strip().lower(),
            "expected_venues": [venue.strip().lower() for venue in expected_venues if venue.strip()],
            "position_tol_base": position_tol_base,
            "max_order_size": max_order_size,
            "telemetry_only_order_path_coverage": True,
            "authority_decision_manifest": False,
            "promotion_evidence": False,
            "fill_evidence_required": False,
        },
        "inputs": {
            "telemetry": _file_meta(telemetry_path),
            "preflight": _file_meta(preflight_path),
            "summary": _file_meta(summary_path),
            "venue_audit": _file_meta(venue_audit_path),
            "run_token": token,
            "audit_captured_after_run_confirmed": audit_captured_after_run,
        },
        "summary": {
            "ticks_run": summary.get("ticks_run"),
            "run_duration_ms": summary.get("run_duration_ms"),
            "kill_events": summary.get("kill_events"),
            "would_place_by_purpose": summary.get("would_place_by_purpose"),
            "would_cancel_by_purpose": summary.get("would_cancel_by_purpose"),
            "would_replace_by_purpose": summary.get("would_replace_by_purpose"),
        },
        "coverage": counts.to_json(),
        "terminal_flatness": {
            "direct_venue_audit_gate_status": "PASS" if not venue_reasons else "HOLD",
            "venue_results": venue_rows,
        },
        "governance": {
            "approved_for_promotion": False,
            "approved_for_live": False,
            "approved_for_capital_escalation": False,
            "blocker_cleared": False,
            "pressure_complete_claim": False,
            "v2_authority_admission": False,
            "authority_manifest_required_for_promotion": True,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--venue-audit", type=Path, required=True)
    parser.add_argument("--target-venue", required=True)
    parser.add_argument("--expected-venues", required=True)
    parser.add_argument("--position-tol-base", type=float, required=True)
    parser.add_argument("--max-order-size", type=float)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--audit-captured-after-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        report = validate_order_path_coverage(
            telemetry_path=args.telemetry,
            preflight_path=args.preflight,
            summary_path=args.summary,
            venue_audit_path=args.venue_audit,
            target_venue=args.target_venue,
            expected_venues=args.expected_venues.split(","),
            position_tol_base=args.position_tol_base,
            max_order_size=args.max_order_size,
            run_token=args.run_token,
            audit_captured_after_run=args.audit_captured_after_run,
        )
    except V2TelemetryCoverageError as exc:
        print(f"V2_TELEMETRY_ORDER_PATH_COVERAGE_ERROR: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["validation_status"] != "PASS":
        print(
            "V2_TELEMETRY_ORDER_PATH_COVERAGE_HOLD "
            f"reasons={','.join(report['validation_reasons'])}"
        )
        return 1
    print("V2_TELEMETRY_ORDER_PATH_COVERAGE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
