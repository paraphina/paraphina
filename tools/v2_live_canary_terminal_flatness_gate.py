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
import re
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


def _load_order_path_coverages(paths: list[Path]) -> dict[str, dict[str, Any]]:
    coverages: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise TerminalFlatnessGateError(f"order-path coverage artifact missing: {path}")
        data = _load_json(path)
        if data.get("artifact_type") != "v2_telemetry_order_path_coverage_manifest":
            raise TerminalFlatnessGateError(f"order-path coverage artifact wrong type: {path}")
        scope = data.get("scope")
        if not isinstance(scope, dict):
            raise TerminalFlatnessGateError(f"order-path coverage artifact missing scope: {path}")
        venue = scope.get("target_venue")
        if not isinstance(venue, str) or not venue.strip():
            raise TerminalFlatnessGateError(f"order-path coverage artifact missing target venue: {path}")
        normalized = venue.strip().lower()
        if normalized in coverages:
            raise TerminalFlatnessGateError(f"duplicate order-path coverage artifact: {normalized}")
        coverages[normalized] = data
    return coverages


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


def _csv_venues(value: str) -> list[str]:
    return [venue.strip().lower() for venue in value.split(",") if venue.strip()]


def _coverage_int(coverage: dict[str, Any], key: str) -> int | None:
    value = coverage.get(key)
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _order_path_coverage_proves_no_exposure(data: dict[str, Any]) -> bool:
    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        return False
    zero_fields = (
        "target_place_intents",
        "target_place_acks",
        "target_cancel_intents",
        "target_cancel_acks",
        "target_would_send_places",
        "target_would_send_cancels",
        "target_replace_events",
        "target_fills",
        "bad_place_policy_count",
    )
    if any(_coverage_int(coverage, key) != 0 for key in zero_fields):
        return False
    observed_actions = coverage.get("observed_order_actions")
    observed_statuses = coverage.get("observed_order_statuses")
    if observed_actions not in ([], None):
        return False
    if observed_statuses not in ([], None):
        return False
    governance = data.get("governance")
    if not isinstance(governance, dict):
        return False
    if governance.get("blocker_cleared") is True or governance.get("pressure_complete_claim") is True:
        return False
    scope = data.get("scope")
    if not isinstance(scope, dict):
        return False
    return scope.get("telemetry_only_order_path_coverage") is True


def _manifest_run_binding_reasons(
    *,
    canary_manifest_path: Path | None,
    venue_audit_path: Path,
    run_token: str | None,
    audit_captured_after_run: bool,
    audit_captured_after_terminal_cleanup: bool,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    manifest_meta: dict[str, Any] = {
        "canary_manifest_path": str(canary_manifest_path) if canary_manifest_path else None,
        "canary_manifest_sha256": None,
        "run_token": run_token,
        "audit_captured_after_run_confirmed": audit_captured_after_run,
        "audit_captured_after_terminal_cleanup_confirmed": audit_captured_after_terminal_cleanup,
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
    if not audit_captured_after_terminal_cleanup:
        reasons.append("run_binding_after_terminal_cleanup_confirmation_missing")
    return reasons, manifest_meta


def _terminal_cleanup_report(
    *,
    live_stderr_path: Path | None,
    promotion_cleanup_strict: bool,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    report: dict[str, Any] = {
        "promotion_cleanup_strict": promotion_cleanup_strict,
        "live_stderr_path": str(live_stderr_path) if live_stderr_path else None,
        "live_stderr_sha256": None,
        "terminal_cancel_timeout_count": 0,
        "terminal_cancel_timeout_ticks": [],
        "terminal_cancel_all_incomplete_count": 0,
        "terminal_cancel_all_incomplete_venues": [],
        "cancel_all_incomplete_direct_venue_audit_cleared": False,
        "hyperliquid_cancel_all_post_inflight_max": 0,
        "hyperliquid_cancel_all_ws_post_count": 0,
        "account_refresh_not_fresh_count": 0,
        "blocked_account_truth_count": 0,
        "blocked_pre_clean_account_truth_count": 0,
        "blocked_post_dispatch_account_truth_count": 0,
        "blocked_final_account_truth_count": 0,
        "pre_clean_account_truth_direct_venue_audit_cleared": False,
        "incomplete_residual_count": 0,
        "clean_terminal_closeout_count": 0,
        "final_check_account_truth": {
            "found": False,
            "requested_venues": [],
            "fresh_venues": [],
            "missing_fresh_venues": [],
            "remaining_venues": [],
            "fresh_covers_requested": False,
            "remaining_venues_empty": False,
            "clean_after_final_seen_after": False,
            "after_latest_account_refresh_not_fresh": False,
        },
        "final_account_truth_no_exposure_direct_venue_audit_cleared": False,
        "final_account_truth_no_exposure_venues": [],
        "final_account_truth_no_exposure_coverage_paths": {},
    }
    if live_stderr_path is None:
        if promotion_cleanup_strict:
            reasons.append("terminal_cleanup_log_missing")
        return reasons, report
    if not live_stderr_path.exists():
        if promotion_cleanup_strict:
            reasons.append("terminal_cleanup_log_missing")
        return reasons, report

    report["live_stderr_sha256"] = _sha256(live_stderr_path)
    text = live_stderr_path.read_text(encoding="utf-8", errors="replace")
    timeout_ticks: list[int] = []
    for match in re.finditer(
        r"tick=(\d+)\s+terminal_exit_cancel:\s+timeout after \d+ms waiting for response",
        text,
    ):
        timeout_ticks.append(int(match.group(1)))
    report["terminal_cancel_timeout_count"] = len(timeout_ticks)
    report["terminal_cancel_timeout_ticks"] = timeout_ticks
    cancel_all_incomplete_venues: list[str] = []
    for match in re.finditer(
        r"canary_exit_cancel_all_cleanup incomplete tracked_open_orders=\d+ venues=(\S*)",
        text,
    ):
        cancel_all_incomplete_venues.extend(_csv_venues(match.group(1)))
    report["terminal_cancel_all_incomplete_count"] = len(cancel_all_incomplete_venues)
    report["terminal_cancel_all_incomplete_venues"] = sorted(set(cancel_all_incomplete_venues))

    post_inflight_max = 0
    ws_cancel_all_count = 0
    for match in re.finditer(
        r"HL_POST_SUBMIT\s+submit_path=ws_post\s+post_id=\d+\s+action_label=cancel_all"
        r"\s+batch_kind=cancel_all\s+batch_size=\d+\s+post_inflight=(\d+)",
        text,
    ):
        ws_cancel_all_count += 1
        post_inflight_max = max(post_inflight_max, int(match.group(1)))
    report["hyperliquid_cancel_all_post_inflight_max"] = post_inflight_max
    report["hyperliquid_cancel_all_ws_post_count"] = ws_cancel_all_count
    account_refresh_not_fresh_matches = list(
        re.finditer(r"canary_exit_position_flatten_cleanup account_refresh_not_fresh", text)
    )
    report["account_refresh_not_fresh_count"] = len(account_refresh_not_fresh_matches)
    latest_account_refresh_not_fresh_end = (
        account_refresh_not_fresh_matches[-1].end() if account_refresh_not_fresh_matches else -1
    )
    report["blocked_pre_clean_account_truth_count"] = text.count(
        "canary_exit_position_flatten_cleanup blocked_pre_clean_account_truth"
    )
    report["blocked_post_dispatch_account_truth_count"] = text.count(
        "canary_exit_position_flatten_cleanup blocked_post_dispatch_account_truth"
    )
    report["blocked_final_account_truth_count"] = text.count(
        "canary_exit_position_flatten_cleanup blocked_final_account_truth"
    )
    report["blocked_account_truth_count"] = (
        report["blocked_pre_clean_account_truth_count"]
        + report["blocked_post_dispatch_account_truth_count"]
        + report["blocked_final_account_truth_count"]
    )
    report["incomplete_residual_count"] = text.count(
        "canary_exit_position_flatten_cleanup incomplete residual_venues="
    )
    report["clean_terminal_closeout_count"] = sum(
        text.count(marker)
        for marker in (
            "canary_exit_position_flatten_cleanup clean_after_account_truth_check",
            "canary_exit_position_flatten_cleanup clean_after_account_refresh",
            "canary_exit_position_flatten_cleanup clean_after_dispatch_account_truth_check",
            "canary_exit_position_flatten_cleanup clean_after_final_account_refresh",
            "canary_exit_cancel_all_cleanup clean",
        )
    )

    final_check_matches = list(
        re.finditer(
            r"canary_exit_position_flatten_cleanup account_refresh_applied "
            r"phase=final_check\s+requested_venues=(\S*)\s+fresh_venues=(\S*)\s+"
            r"position_changed=\S+\s+remaining_venues=(\S*)",
            text,
        )
    )
    if final_check_matches:
        latest = final_check_matches[-1]
        requested_venues = _csv_venues(latest.group(1))
        fresh_venues = _csv_venues(latest.group(2))
        remaining_venues = _csv_venues(latest.group(3))
        clean_marker_index = text.find(
            "canary_exit_position_flatten_cleanup clean_after_final_account_refresh",
            latest.end(),
        )
        final_check_after_latest_not_fresh = latest.start() > latest_account_refresh_not_fresh_end
        report["final_check_account_truth"] = {
            "found": True,
            "requested_venues": requested_venues,
            "fresh_venues": fresh_venues,
            "missing_fresh_venues": sorted(set(requested_venues).difference(fresh_venues)),
            "remaining_venues": remaining_venues,
            "fresh_covers_requested": set(requested_venues).issubset(set(fresh_venues)),
            "remaining_venues_empty": len(remaining_venues) == 0,
            "clean_after_final_seen_after": clean_marker_index >= 0,
            "after_latest_account_refresh_not_fresh": final_check_after_latest_not_fresh,
        }

    if promotion_cleanup_strict:
        if timeout_ticks:
            reasons.append("terminal_cleanup_cancel_timeout_present")
        if post_inflight_max > 1:
            reasons.append("terminal_cleanup_hyperliquid_cancel_all_post_inflight_backlog")
        if report["incomplete_residual_count"] > 0:
            reasons.append("terminal_cleanup_incomplete_residual_present")
        if (
            report["blocked_post_dispatch_account_truth_count"] > 0
            or report["blocked_final_account_truth_count"] > 0
        ):
            reasons.append("terminal_cleanup_account_truth_blocked")
        final_check_account_truth = report["final_check_account_truth"]
        account_refresh_not_fresh_cleared = (
            final_check_account_truth["found"]
            and len(final_check_account_truth["requested_venues"]) > 0
            and final_check_account_truth["fresh_covers_requested"]
            and final_check_account_truth["remaining_venues_empty"]
            and final_check_account_truth["clean_after_final_seen_after"]
            and final_check_account_truth["after_latest_account_refresh_not_fresh"]
        )
        if (
            report["account_refresh_not_fresh_count"] > 0
            and report["blocked_account_truth_count"] == 0
            and report["incomplete_residual_count"] == 0
            and not account_refresh_not_fresh_cleared
        ):
            reasons.append("terminal_cleanup_account_refresh_not_fresh")
    return reasons, report


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
    audit_captured_after_terminal_cleanup: bool = False,
    live_stderr_path: Path | None = None,
    promotion_cleanup_strict: bool = False,
    order_path_coverage_paths: list[Path] | None = None,
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
        audit_captured_after_terminal_cleanup=audit_captured_after_terminal_cleanup,
    )
    reasons.extend(binding_reasons)
    cleanup_reasons, cleanup_report = _terminal_cleanup_report(
        live_stderr_path=live_stderr_path,
        promotion_cleanup_strict=promotion_cleanup_strict,
    )
    reasons.extend(cleanup_reasons)
    order_path_coverages = _load_order_path_coverages(order_path_coverage_paths or [])
    order_path_coverage_paths_by_venue = {
        data["scope"]["target_venue"].strip().lower(): path
        for path in (order_path_coverage_paths or [])
        for data in [_load_json(path)]
        if isinstance(data.get("scope"), dict)
        and isinstance(data["scope"].get("target_venue"), str)
    }
    direct_venue_reasons: list[str] = []

    if data.get("ok") is not True:
        direct_venue_reasons.append("venue_audit_not_ok")
    if _as_int(data.get("max_open_orders")) != 0:
        direct_venue_reasons.append("venue_audit_max_open_orders_not_zero")
    if data.get("allow_unknown_open_orders") is True:
        direct_venue_reasons.append("venue_audit_allows_unknown_open_orders")

    audit_tol = _as_float(data.get("position_tol_base"))
    if audit_tol is None:
        direct_venue_reasons.append("venue_audit_missing_position_tolerance")
    elif abs(audit_tol - position_tol_base) > 1e-12:
        direct_venue_reasons.append("venue_audit_position_tolerance_mismatch")

    for venue in expected:
        result = observed.get(venue)
        if result is None:
            direct_venue_reasons.append(f"{venue}:missing_venue_result")
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
            direct_venue_reasons.extend(f"{venue}:{reason}" for reason in row_reasons)
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

    reasons.extend(direct_venue_reasons)
    direct_venue_status = "PASS" if not direct_venue_reasons else "HOLD"
    run_binding_status = "PASS" if not binding_reasons else "HOLD"
    if (
        promotion_cleanup_strict
        and cleanup_report["blocked_pre_clean_account_truth_count"] > 0
        and cleanup_report["blocked_post_dispatch_account_truth_count"] == 0
        and cleanup_report["blocked_final_account_truth_count"] == 0
    ):
        if direct_venue_status == "PASS" and run_binding_status == "PASS":
            cleanup_report["pre_clean_account_truth_direct_venue_audit_cleared"] = True
        else:
            cleanup_reasons.append("terminal_cleanup_account_truth_blocked")
            reasons.append("terminal_cleanup_account_truth_blocked")
    if promotion_cleanup_strict and cleanup_report["terminal_cancel_all_incomplete_count"] > 0:
        if direct_venue_status == "PASS" and run_binding_status == "PASS":
            cleanup_report["cancel_all_incomplete_direct_venue_audit_cleared"] = True
        else:
            cleanup_reasons.append("terminal_cleanup_cancel_all_incomplete")
            reasons.append("terminal_cleanup_cancel_all_incomplete")
    if (
        promotion_cleanup_strict
        and cleanup_report["blocked_final_account_truth_count"] > 0
        and cleanup_report["blocked_post_dispatch_account_truth_count"] == 0
        and cleanup_report["incomplete_residual_count"] == 0
        and direct_venue_status == "PASS"
        and run_binding_status == "PASS"
    ):
        final_check = cleanup_report["final_check_account_truth"]
        missing_fresh_venues = list(final_check.get("missing_fresh_venues") or [])
        if (
            final_check.get("found") is True
            and final_check.get("remaining_venues_empty") is True
            and missing_fresh_venues
            and all(
                venue in order_path_coverages
                and _order_path_coverage_proves_no_exposure(order_path_coverages[venue])
                for venue in missing_fresh_venues
            )
        ):
            cleanup_report["final_account_truth_no_exposure_direct_venue_audit_cleared"] = True
            cleanup_report["final_account_truth_no_exposure_venues"] = sorted(missing_fresh_venues)
            cleanup_report["final_account_truth_no_exposure_coverage_paths"] = {
                venue: str(order_path_coverage_paths_by_venue[venue])
                for venue in sorted(missing_fresh_venues)
                if venue in order_path_coverage_paths_by_venue
            }
            cleanup_reasons = [
                reason for reason in cleanup_reasons if reason != "terminal_cleanup_account_truth_blocked"
            ]
            reasons = [reason for reason in reasons if reason != "terminal_cleanup_account_truth_blocked"]
    promotion_cleanup_strict_status = "PASS" if not cleanup_reasons else "HOLD"
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
        "terminal_cleanup": cleanup_report,
        "closeout_status": {
            "direct_venue_audit_status": direct_venue_status,
            "direct_venue_audit_reasons": direct_venue_reasons,
            "promotion_cleanup_strict_status": promotion_cleanup_strict_status,
            "promotion_cleanup_strict_reasons": cleanup_reasons,
            "run_binding_status": run_binding_status,
            "run_binding_reasons": binding_reasons,
            "promotion_ready": status == "PASS",
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
    parser.add_argument(
        "--audit-captured-after-terminal-cleanup",
        action="store_true",
        help="Required explicit confirmation that the venue audit was captured after terminal cleanup completed.",
    )
    parser.add_argument("--expected-venues", required=True)
    parser.add_argument("--position-tol-base", type=float, required=True)
    parser.add_argument(
        "--live-stderr",
        type=Path,
        help="Saved live stderr artifact. Required for promotion cleanup strictness.",
    )
    parser.add_argument(
        "--promotion-cleanup-strict",
        action="store_true",
        help="HOLD if terminal cleanup logs contain timeout/account-truth promotion gaps.",
    )
    parser.add_argument(
        "--order-path-coverage",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional telemetry order-path coverage manifest. In strict mode, this may clear a "
            "final account-truth freshness gap only for venues with zero order/fill exposure and "
            "clean direct venue audit."
        ),
    )
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
            audit_captured_after_terminal_cleanup=args.audit_captured_after_terminal_cleanup,
            live_stderr_path=args.live_stderr,
            promotion_cleanup_strict=args.promotion_cleanup_strict,
            order_path_coverage_paths=args.order_path_coverage,
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
