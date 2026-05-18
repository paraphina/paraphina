#!/usr/bin/env python3
"""Phase 5.1ak non-live blocker-resolution runner.

This HOLD-only runner stitches the approved current-pack recovery path into one
repo-owned gate:

- optionally normalize directly target-linkable private/native rows through
  Phase 5.1aj;
- optionally stage sanitized Lighter event-time pressure rows through
  Phase 5.1ab;
- compose all supplied manifests through Phase 5.1ae;
- validate the composed manifest through Phase 5.1v;
- emit a target-level recovery/forward-refresh decision artifact.

It performs no network access, reads no env files, places no orders, cancels no
orders, does not call sendTx/sendTxBatch, and never infers source links from
time, price, size, venue role, or proximity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase51ab_lighter_native_limit_pressure_source import build_lighter_native_limit_pressure_source
from phase51ae_candidate_manifest_compose import build_candidate_manifest_composition
from phase51aj_forward_private_stream_source import build_forward_private_stream_source
from phase51v_forward_capture_bundle_readiness import build_forward_capture_bundle_readiness


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ak_blocker_resolution_runner"
DEFAULT_TARGET_RUN = (
    ROOT
    / "runs/phase51u_forward_capture_target_manifest"
    / "PHASE51U-FORWARD-CAPTURE-TARGET-LINK-HYGIENE-20260505T000000Z"
)
DEFAULT_REQUEST_PACK = (
    ROOT
    / "runs/phase51z_source_link_request_pack"
    / "PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z"
)
DEFAULT_CURRENT_CANDIDATE_MANIFEST = (
    ROOT
    / "runs/phase51ae_candidate_manifest_compose"
    / "PHASE51AE-BLOCKER-RECHECK-PLUS-PHASE51AJ-HOLD-20260506T000000Z"
    / "candidate_manifest.composed.json"
)
SOURCE_OWNER_NATIVE_ROLE_READY_HI_DEFERRED_STATUS = "SOURCE_OWNER_NATIVE_ROLE_READY_HI_DEFERRED"
SOURCE_OWNER_NATIVE_ROLE_INCOMPLETE_STATUS = "SOURCE_OWNER_NATIVE_ROLE_INCOMPLETE"
SOURCE_OWNER_NATIVE_ROLE_READY_PRESSURE_PENDING_STATUS = "SOURCE_OWNER_NATIVE_ROLE_READY_PRESSURE_PENDING"
UNSAFE_TRUE_FLAGS = {
    "approved_for_model_training",
    "approved_for_live",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_financial_claim",
    "admissible_for_model_training",
    "admissible_for_financial_claim",
    "admissible_for_ev_admission",
    "live_orders_allowed",
    "capital_change_allowed",
    "risk_limit_relaxation_allowed",
}
SECRET_FIELD_FRAGMENTS = {
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "session_token",
    "signing_key",
}
RAW_IDENTIFIER_FIELDS = {
    "ask_client_id",
    "ask_id",
    "bid_client_id",
    "bid_id",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "decision_id",
    "fill_id",
    "fillId",
    "i",
    "id",
    "oid",
    "order_id",
    "orderId",
    "raw_client_order_id",
    "raw_order_id",
    "tid",
    "trade_id",
    "tradeId",
    "tx_hash",
    "txHash",
    "venue_order_id",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield line_no, row


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_output_safe(data, path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            _check_output_safe(record, path)
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _field_looks_secret(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if "nonsecret" in normalized:
        return False
    if "authorization" in normalized and normalized.startswith(("no_", "not_")):
        return False
    return any(fragment in normalized for fragment in SECRET_FIELD_FRAGMENTS)


def _check_no_secret_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_unsafe_flags(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_no_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = sorted(RAW_IDENTIFIER_FIELDS & set(obj))
        if raw_fields:
            raise ValueError(f"{path} {label} leaked raw identifier fields: {raw_fields}")


def _check_output_safe(record: Any, path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


def _target_id(row: dict[str, Any]) -> str:
    return str(row.get("canonical_group_id") or row.get("order_key") or "")


def _load_targets(target_run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_path = target_run / "native_role_capture_targets.jsonl"
    limit_path = target_run / "lighter_native_limit_capture_targets.jsonl"
    if not role_path.exists() or not limit_path.exists():
        raise ValueError(f"target run missing Phase 5.1u target files: {target_run}")
    role_targets = [row for _, row in _iter_jsonl(role_path)]
    limit_targets = [row for _, row in _iter_jsonl(limit_path)]
    return role_targets, limit_targets


def _target_ref(row: dict[str, Any], target_type: str) -> dict[str, Any]:
    group = str(row.get("canonical_group_id") or "")
    order_key = str(row.get("order_key") or "")
    venue = str(row.get("venue_id") or "unknown").lower()
    return {
        "target_type": target_type,
        "venue_id": venue,
        "canonical_group_id": group,
        "order_key": order_key,
        "target_ref_sha256": _stable_hash(
            {
                "target_type": target_type,
                "venue_id": venue,
                "canonical_group_id": group,
                "order_key": order_key,
            }
        ),
    }


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _decision_counts_by_type_venue(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = ":".join(
            [
                str(record.get("target_type") or "unknown"),
                str(record.get("venue_id") or "unknown"),
                str(record.get("decision_status") or "UNKNOWN"),
            ]
        )
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _source_owner_scope(
    *,
    native_role_target_count: int,
    missing_role: int,
    missing_limit: int,
    pressure_unavailable_source_count: int,
    pressure_unavailable_targets: int,
) -> dict[str, Any]:
    native_role_ready = native_role_target_count > 0 and missing_role == 0
    pressure_unavailable_observed = pressure_unavailable_source_count > 0
    pressure_unavailable_accounts_for_limits = (
        pressure_unavailable_targets > 0 and missing_limit == pressure_unavailable_targets
    )
    limit_dependency_satisfied = missing_limit == 0 or pressure_unavailable_accounts_for_limits
    source_owner_ready = native_role_ready and limit_dependency_satisfied
    if source_owner_ready:
        status = SOURCE_OWNER_NATIVE_ROLE_READY_HI_DEFERRED_STATUS
        next_required_action = "record_scoped_source_owner_native_role_acceptance_and_defer_h_i_calibration"
    elif native_role_ready:
        status = SOURCE_OWNER_NATIVE_ROLE_READY_PRESSURE_PENDING_STATUS
        next_required_action = "complete_or_govern_lighter_native_limit_pressure_before_scoped_acceptance"
    else:
        status = SOURCE_OWNER_NATIVE_ROLE_INCOMPLETE_STATUS
        next_required_action = "obtain_target_linked_native_role_source_owner_evidence"
    return {
        "source_owner_native_role_evidence_ready": source_owner_ready,
        "source_owner_native_role_ready_without_h_i": source_owner_ready,
        "phase51_source_owner_blocker_status": status,
        "lighter_pressure_unavailable_governance_accepted": native_role_ready and pressure_unavailable_observed,
        "h_i_feature_matrix_deferred": source_owner_ready,
        "h_i_feature_matrix_deferred_reason": (
            "source_owner_native_role_scope_does_not_require_pfill_feature_matrix"
            if source_owner_ready
            else None
        ),
        "phase51_global_blocker_status": "HOLD",
        "source_owner_scope_next_required_action": next_required_action,
    }


def _decision_rows(
    *,
    run_id: str,
    timestamp_ns: int,
    target_run: Path,
    phase51v_run: Path,
    phase51v_summary: dict[str, Any],
    target_pack_mode: str,
) -> list[dict[str, Any]]:
    role_targets, limit_targets = _load_targets(target_run)
    missing_role_ids = {
        _target_id(row)
        for _, row in _iter_jsonl(phase51v_run / "missing_native_role_capture_targets.jsonl")
    }
    missing_limit_ids = {
        _target_id(row)
        for _, row in _iter_jsonl(phase51v_run / "missing_lighter_native_limit_capture_targets.jsonl")
    }
    unavailable_limit_targets = int(phase51v_summary.get("lighter_native_limit_pressure_unavailable_target_count") or 0)
    pressure_unavailable_active = unavailable_limit_targets > 0

    rows: list[dict[str, Any]] = []
    seq = 0
    for target_type, targets, missing_ids in (
        ("native_role", role_targets, missing_role_ids),
        ("lighter_native_limit", limit_targets, missing_limit_ids),
    ):
        for target in targets:
            target_id = _target_id(target)
            missing = target_id in missing_ids
            if target_type == "lighter_native_limit" and missing and pressure_unavailable_active:
                decision_status = "PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD"
                next_required_action = "APPLY_REVISED_PRESSURE_UNAVAILABLE_GOVERNANCE_CONTRACT"
                forward_refresh_required = False
            elif target_pack_mode == "forward-refresh":
                decision_status = "FORWARD_REFRESH_PACK_INCOMPLETE" if missing else "READY_FORWARD_REFRESH_PACK"
                next_required_action = "FORWARD_REFRESH_SOURCE_TRUTH_REQUIRED" if missing else "NONE"
                forward_refresh_required = missing
            else:
                decision_status = "UNRECOVERABLE_FROM_LOCAL_ARTIFACTS" if missing else "RECOVERED_CURRENT_PACK"
                next_required_action = "FORWARD_REFRESH_REQUIRED" if missing else "NONE"
                forward_refresh_required = missing
            row = {
                "schema_version": 1,
                "label_type": "PHASE51AK_BLOCKER_TARGET_DECISION",
                "label_seq": seq,
                "timestamp_local_ns": timestamp_ns + seq,
                "timestamp_utc": _timestamp_ns_to_utc(timestamp_ns),
                "run_id": run_id,
                "baseline_commit": BASELINE_COMMIT,
                "gate_status": "HOLD",
                "decision_status": decision_status,
                "current_pack_target_ready": not missing,
                "current_pack_missing_after_final_validation": missing,
                "next_required_action": next_required_action,
                "forward_refresh_required": forward_refresh_required,
                "pressure_unavailable_governance_hold": (
                    target_type == "lighter_native_limit" and missing and pressure_unavailable_active
                ),
                "target_pack_mode": target_pack_mode,
                "phase51v_run": str(phase51v_run),
                "no_live_flag": True,
                "approved_for_model_training": False,
                "approved_for_live": False,
                "approved_for_canary": False,
                "approved_for_capital_escalation": False,
                "admissible_for_financial_claim": False,
                "admissible_for_ev_admission": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
                "raw_identifier_redaction_status": "PASS",
            }
            row.update(_target_ref(target, target_type))
            rows.append(row)
            seq += 1
    return rows


def _component_info(name: str, path: Path, summary_name: str | None) -> dict[str, Any]:
    info = {
        "component": name,
        "path": str(path),
        "manifest_sha256": _sha256_file(path / "manifest.json") if (path / "manifest.json").exists() else None,
    }
    if summary_name:
        summary_path = path / summary_name
        info["summary_path"] = str(summary_path)
        info["summary_sha256"] = _sha256_file(summary_path) if summary_path.exists() else None
    return info


def build_blocker_resolution_runner(
    *,
    target_run: Path,
    request_pack: Path,
    candidate_manifests: list[Path],
    use_default_current_manifest: bool,
    phase51aj_source_specs: list[str],
    phase51ab_pressure_jsonl: Path | None,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
    target_pack_mode: str,
) -> Path:
    run_id = _check_run_id(run_id)
    target_run = _resolve_path(target_run)
    request_pack = _resolve_path(request_pack)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    component_root = out_dir / "component_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not target_run.exists() or not target_run.is_dir():
        raise ValueError(f"target-run path must be an existing directory: {target_run}")
    if not (request_pack / "manifest.json").exists():
        raise ValueError(f"request-pack path must contain manifest.json: {request_pack}")

    component_infos: list[dict[str, Any]] = []
    manifests_for_compose: list[Path] = []
    if use_default_current_manifest:
        manifests_for_compose.append(DEFAULT_CURRENT_CANDIDATE_MANIFEST)
    manifests_for_compose.extend(_resolve_path(path) for path in candidate_manifests)

    if phase51aj_source_specs:
        phase51aj_run = build_forward_private_stream_source(
            request_pack=request_pack,
            output_root=component_root / "phase51aj_forward_private_stream_source",
            run_id=f"{run_id}-PHASE51AJ-HOLD",
            timestamp_ns=timestamp_ns,
            source_specs=phase51aj_source_specs,
        )
        manifests_for_compose.append(phase51aj_run / "phase51aj_candidate_manifest.json")
        component_infos.append(
            _component_info(
                "phase51aj_forward_private_stream_source",
                phase51aj_run,
                "phase51aj_forward_private_stream_source_summary.json",
            )
        )

    if phase51ab_pressure_jsonl is not None:
        phase51ab_run = build_lighter_native_limit_pressure_source(
            input_jsonl=_resolve_path(phase51ab_pressure_jsonl),
            target_run=target_run,
            output_root=component_root / "phase51ab_lighter_native_limit_pressure_source",
            run_id=f"{run_id}-PHASE51AB-HOLD",
            timestamp_ns=timestamp_ns,
        )
        manifests_for_compose.append(phase51ab_run / "phase51v_candidate_manifest.json")
        component_infos.append(
            _component_info(
                "phase51ab_lighter_native_limit_pressure_source",
                phase51ab_run,
                "phase51ab_lighter_native_limit_pressure_summary.json",
            )
        )

    if not manifests_for_compose:
        raise ValueError("no candidate manifests supplied; provide --candidate-manifest or generated source inputs")

    phase51ae_run = build_candidate_manifest_composition(
        candidate_manifests=manifests_for_compose,
        source_specs=[],
        source_link_specs=[],
        output_root=component_root / "phase51ae_candidate_manifest_compose",
        run_id=f"{run_id}-PHASE51AE-HOLD",
        timestamp_ns=timestamp_ns,
        target_run=target_run,
    )
    component_infos.append(
        _component_info(
            "phase51ae_candidate_manifest_compose",
            phase51ae_run,
            "phase51ae_candidate_manifest_compose_summary.json",
        )
    )

    phase51v_run = build_forward_capture_bundle_readiness(
        target_run=target_run,
        candidate_manifest=phase51ae_run / "candidate_manifest.composed.json",
        output_root=component_root / "phase51v_forward_capture_bundle_readiness",
        run_id=f"{run_id}-PHASE51V-HOLD",
        timestamp_ns=timestamp_ns,
    )
    component_infos.append(
        _component_info(
            "phase51v_forward_capture_bundle_readiness",
            phase51v_run,
            "phase51v_forward_capture_bundle_readiness_summary.json",
        )
    )

    phase51v_summary_path = phase51v_run / "phase51v_forward_capture_bundle_readiness_summary.json"
    phase51v_summary = _load_json(phase51v_summary_path)
    decision_rows = _decision_rows(
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        target_run=target_run,
        phase51v_run=phase51v_run,
        phase51v_summary=phase51v_summary,
        target_pack_mode=target_pack_mode,
    )
    decision_path = out_dir / "phase51ak_blocker_target_decisions.jsonl"
    summary_path = out_dir / "phase51ak_blocker_resolution_summary.json"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(decision_path, decision_rows)
    missing_role = int(phase51v_summary.get("native_role_capture_target_missing_count") or 0)
    native_role_target_count = int(phase51v_summary.get("native_role_capture_target_count") or 0)
    missing_limit = int(phase51v_summary.get("lighter_native_limit_capture_target_missing_count") or 0)
    pressure_unavailable_targets = int(
        phase51v_summary.get("lighter_native_limit_pressure_unavailable_target_count") or 0
    )
    pressure_unavailable_source_count = int(
        phase51v_summary.get("lighter_native_limit_pressure_unavailable_source_count") or 0
    )
    downstream_ready = bool(phase51v_summary.get("downstream_chain_ready"))
    pressure_unavailable_governance_hold = pressure_unavailable_targets > 0
    unresolved_without_governance = missing_role > 0 or (missing_limit - pressure_unavailable_targets) > 0
    source_owner_scope = _source_owner_scope(
        native_role_target_count=native_role_target_count,
        missing_role=missing_role,
        missing_limit=missing_limit,
        pressure_unavailable_source_count=pressure_unavailable_source_count,
        pressure_unavailable_targets=pressure_unavailable_targets,
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": (
            "phase51ak_source_owner_native_role_ready_hi_deferred_nonlive_hold"
            if source_owner_scope["source_owner_native_role_evidence_ready"]
            else
            (
                "phase51ak_forward_refresh_pack_ready_nonlive_hold"
                if target_pack_mode == "forward-refresh"
                else "phase51ak_current_pack_ready_nonlive_hold"
            )
            if downstream_ready
            else (
                "phase51ak_pressure_unavailable_governance_hold_nonlive"
                if pressure_unavailable_governance_hold and not unresolved_without_governance
                else (
                    "phase51ak_forward_refresh_pack_incomplete_nonlive_hold"
                    if target_pack_mode == "forward-refresh"
                    else "phase51ak_current_pack_incomplete_forward_refresh_required_nonlive_hold"
                )
            )
        ),
        "target_pack_mode": target_pack_mode,
        "target_run": str(target_run),
        "request_pack": str(request_pack),
        "candidate_manifest_count": len(manifests_for_compose),
        "phase51aj_source_spec_count": len(phase51aj_source_specs),
        "phase51ab_pressure_source_supplied": phase51ab_pressure_jsonl is not None,
        "phase51ae_run": str(phase51ae_run),
        "phase51v_run": str(phase51v_run),
        "phase51v_summary_path": str(phase51v_summary_path),
        "native_role_capture_target_count": phase51v_summary.get("native_role_capture_target_count"),
        "native_role_capture_target_ready_count": phase51v_summary.get("native_role_capture_target_ready_count"),
        "native_role_capture_target_missing_count": missing_role,
        **source_owner_scope,
        "lighter_native_limit_capture_target_count": phase51v_summary.get("lighter_native_limit_capture_target_count"),
        "lighter_native_limit_capture_target_ready_count": phase51v_summary.get(
            "lighter_native_limit_capture_target_ready_count"
        ),
        "lighter_native_limit_capture_target_missing_count": missing_limit,
        "lighter_native_limit_pressure_unavailable_target_count": pressure_unavailable_targets,
        "pressure_unavailable_governance_hold": pressure_unavailable_governance_hold,
        "revised_pressure_unavailable_contract_clears_blocker": False,
        "phase51v_downstream_chain_ready": downstream_ready,
        "decision_row_count": len(decision_rows),
        "decision_status_counts": _status_counts(decision_rows, "decision_status"),
        "decision_status_counts_by_target_type_venue": _decision_counts_by_type_venue(decision_rows),
        "forward_refresh_required": bool(unresolved_without_governance if pressure_unavailable_governance_hold else not downstream_ready),
        "validated_mapping_required_for_current_pack": bool(
            unresolved_without_governance if pressure_unavailable_governance_hold else not downstream_ready
        ),
        "next_required_action": (
            source_owner_scope["source_owner_scope_next_required_action"]
            if source_owner_scope["source_owner_native_role_evidence_ready"]
            else
            "run_phase51s_to_phase51i_nonlive_ladder"
            if downstream_ready
            else (
                "apply_revised_pressure_unavailable_governance_contract"
                if pressure_unavailable_governance_hold and not unresolved_without_governance
                else "obtain_validated_mapping_or_forward_refresh_target_pack_with_event_time_sources"
            )
        ),
        "component_runs": component_infos,
        "clears_phase51_blockers": False,
        "no_live_flag": True,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
    }
    _write_json(summary_path, summary)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": _artifact_infos(out_dir, [decision_path, summary_path]),
            "component_runs": component_infos,
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        },
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, default=DEFAULT_TARGET_RUN)
    parser.add_argument("--request-pack", type=Path, default=DEFAULT_REQUEST_PACK)
    parser.add_argument("--candidate-manifest", action="append", type=Path, default=[])
    parser.add_argument("--no-default-current-manifest", action="store_true")
    parser.add_argument("--phase51aj-source-json", action="append", default=[], help="venue=/path/to/file")
    parser.add_argument("--phase51ab-pressure-jsonl", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AK-BLOCKER-RESOLUTION-RUNNER-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--target-pack-mode", choices=("current-pack", "forward-refresh"), default="current-pack")
    args = parser.parse_args()

    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_blocker_resolution_runner(
            target_run=args.target_run,
            request_pack=args.request_pack,
            candidate_manifests=args.candidate_manifest,
            use_default_current_manifest=not args.no_default_current_manifest,
            phase51aj_source_specs=args.phase51aj_source_json,
            phase51ab_pressure_jsonl=args.phase51ab_pressure_jsonl,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
            target_pack_mode=args.target_pack_mode,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail closed with a concise error
        print(f"phase51ak_blocker_resolution_runner: ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"phase51ak_blocker_resolution_runner: wrote {out_dir}")
    print("phase51ak_blocker_resolution_runner: status HOLD (non-live blocker decision only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
