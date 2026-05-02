#!/usr/bin/env python3
"""Phase 5.1 non-live EV shadow artifact generator.

This tool does not send orders and does not modify runtime services.  It reads
an optional v1 telemetry file, extracts Lighter quote-level candidates, and
emits schema_version=2 EV shadow records that intentionally default to HOLD
until Phase 5.1 calibration evidence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
CALIBRATION_STATUS = "SPARSE"
CALIBRATION_SAMPLE_COUNT = 0
MIN_QUOTE_CANDIDATES_REQUIRED = 1_000
MIN_FILL_LABELS_REQUIRED = 200
MIN_HEDGE_LABELS_REQUIRED = 100
CALIBRATION_HOLD_REASONS = [
    "missing_pfill_calibration",
    "missing_markout_calibration",
    "missing_hedge_success_calibration",
    "missing_queue_reset_calibration",
    "missing_churn_calibration",
    "missing_tail_risk_calibration",
    "sparse_calibration_bucket",
    "counterfactual_only_nonfinancial",
]
ACTION_OWNER_NONLIVE = "NO_ACTION_NONLIVE_SHADOW"
DOUBLE_ACTION_STATE_NONLIVE = "NO_EXECUTION_EVENTS_EMITTED"
FAST_HEDGE_STATE_NONLIVE = "NOT_APPLICABLE_NONLIVE_SHADOW"
RESIDUAL_STATE_STATUS_NONLIVE = "NO_FILL_NO_RESIDUAL"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON in {path}")
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _derive_replay_timestamp_ns(spec: dict[str, Any], run_id: str, input_sha256: str | None) -> int:
    payload = {
        "baseline_commit": BASELINE_COMMIT,
        "experiment_id": spec.get("experiment_id"),
        "input_sha256": input_sha256,
        "run_id": run_id,
    }
    digest = _stable_hash(payload)
    offset_ns = int(digest[:14], 16) % 10_000_000_000_000_000
    return 1_700_000_000_000_000_000 + offset_ns


def _resolve_output_dir(spec: dict[str, Any], output_root: Path | None, run_id: str) -> Path:
    root = output_root or ROOT / spec.get("output_root", "runs/phase51_lighter_only_ev_shadow")
    if not root.is_absolute():
        root = ROOT / root
    return root / run_id


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("spec baseline_commit does not match clean Phase 5 baseline")
    if spec.get("run_mode") != "SHADOW":
        raise ValueError("Phase 5.1 first experiment run_mode must be SHADOW")
    if spec.get("venue_id") != "lighter":
        raise ValueError("Phase 5.1 first experiment spec must be Lighter-only")
    if spec.get("no_live_flag") is not True:
        raise ValueError("no_live_flag must be true")
    if spec.get("capital_escalation_flag") is not False:
        raise ValueError("capital escalation must be false")
    if spec.get("risk_limit_override_flag") is not False:
        raise ValueError("risk limit override must be false")
    constraints = spec.get("constraints", {})
    if constraints.get("live_orders_allowed") is not False:
        raise ValueError("live_orders_allowed must be false")
    if constraints.get("capital_change_allowed") is not False:
        raise ValueError("capital_change_allowed must be false")
    if constraints.get("risk_limit_relaxation_allowed") is not False:
        raise ValueError("risk_limit_relaxation_allowed must be false")


def _base_event(
    *,
    event_type: str,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "event_type": event_type,
        "event_seq": event_seq,
        "timestamp_local_ns": timestamp_ns + event_seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iter_input_records(path: Path) -> tuple[int, list[tuple[int, str, dict[str, Any]]]]:
    records: list[tuple[int, str, dict[str, Any]]] = []
    scanned = 0
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            source = line.strip()
            if not source:
                continue
            pos = 0
            while pos < len(source):
                try:
                    obj, end = decoder.raw_decode(source, pos)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_num}:{pos + 1}: {exc}") from exc
                if not isinstance(obj, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_num}")
                scanned += 1
                record_hash = hashlib.sha256(source[pos:end].encode("utf-8")).hexdigest()
                records.append((line_num, record_hash, obj))
                pos = end
                while pos < len(source) and source[pos].isspace():
                    pos += 1
    return scanned, records


def _quote_candidates_from_record(
    record: dict[str, Any],
    *,
    line_num: int,
    record_hash: str,
    run_id: str,
    venue_id: str,
    instrument_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source_t = int(_number(record.get("t"), 0.0))
    source_schema_version = int(_number(record.get("schema_version"), 1.0))
    for quote_index, quote in enumerate(record.get("quote_levels", []) or []):
        if not isinstance(quote, dict):
            continue
        if quote.get("venue_id") != venue_id:
            continue
        side = quote.get("side")
        price = quote.get("price")
        size = quote.get("size_final")
        if not isinstance(price, (int, float)) or not isinstance(size, (int, float)):
            continue
        side_text = str(side or "UNKNOWN").upper()
        layer = "BASELINE_COMPAT"
        feature_payload = {
            "book_age_ms": quote.get("book_age_ms"),
            "candidate_edge_pre_utility": quote.get("candidate_edge_pre_utility"),
            "distance_to_touch_bps": quote.get("distance_to_touch_bps"),
            "edge_local": quote.get("edge_local"),
            "edge_threshold": quote.get("edge_threshold"),
            "fair_value": record.get("fair_value"),
            "price": price,
            "quote_state": quote.get("quote_state"),
            "side": side_text,
            "size_final": size,
            "source_record_sha256": record_hash,
            "venue_id": venue_id,
        }
        candidate_id = "cand_" + _stable_hash({
            "run_id": run_id,
            "source_line": line_num,
            "source_t": source_t,
            "quote_index": quote_index,
            "feature_payload": feature_payload,
        })[:24]
        candidates.append({
            "candidate_id": candidate_id,
            "source_schema_version": source_schema_version,
            "source_t": source_t,
            "source_line": line_num,
            "source_record_sha256": record_hash,
            "config_version_id": _optional_text(record.get("config_version_id")),
            "instrument_id": instrument_id,
            "venue_id": venue_id,
            "side": side_text,
            "layer": layer,
            "price": float(price),
            "size": max(float(size), 0.0),
            "candidate_notional_usd": max(float(size), 0.0) * float(price),
            "local_edge_feature": _number(quote.get("edge_local")),
            "pair_edge_feature": None,
            "quote_state": _optional_text(quote.get("quote_state")),
            "suppression_reason": _optional_text(quote.get("suppression_reason")),
            "fair_value": _number(record.get("fair_value")),
            "book_age_ms": _number(quote.get("book_age_ms")),
            "book_stale_threshold_ms": _number(quote.get("book_stale_threshold_ms")),
            "distance_to_touch_bps": _number(quote.get("distance_to_touch_bps")),
            "edge_threshold": _number(quote.get("edge_threshold")),
            "candidate_edge_pre_utility": _number(quote.get("candidate_edge_pre_utility")),
            "size_raw": _number(quote.get("size_raw")),
            "size_margin_cap": _number(quote.get("size_margin_cap")),
            "utility_tier": _optional_text(quote.get("utility_tier")),
            "utility_role": _optional_text(quote.get("utility_role")),
            "utility_reason": _optional_text(quote.get("utility_reason")),
            "model_features_hash": _stable_hash(feature_payload),
            "calibration_bucket_id": f"{venue_id}:{side_text}:{layer}:uncalibrated",
            "calibration_status": CALIBRATION_STATUS,
            "calibration_sample_count": CALIBRATION_SAMPLE_COUNT,
            "pair_conditioned_flag": False,
            "fast_hedge_allowed": False,
            "fast_hedge_serialization_state": FAST_HEDGE_STATE_NONLIVE,
            "residual_state_required": False,
            "residual_state_status": RESIDUAL_STATE_STATUS_NONLIVE,
            "action_owner": ACTION_OWNER_NONLIVE,
            "double_action_prevention_state": DOUBLE_ACTION_STATE_NONLIVE,
            "min_quote_candidates_required": MIN_QUOTE_CANDIDATES_REQUIRED,
            "min_fill_labels_required": MIN_FILL_LABELS_REQUIRED,
            "min_hedge_labels_required": MIN_HEDGE_LABELS_REQUIRED,
            "binding_constraints": [
                "phase51_nonlive_hold",
                "missing_phase51_calibration",
                *CALIBRATION_HOLD_REASONS,
            ],
            "decision_reason_primary": "phase51_calibration_hold",
            "decision_reason_secondary_list": CALIBRATION_HOLD_REASONS,
        })
    return candidates


def _ev_event(candidate: dict[str, Any], *, event_seq: int, run_id: str, timestamp_ns: int, alpha: float) -> dict[str, Any]:
    event = _base_event(
        event_type="V2_EV_EVALUATED",
        event_seq=event_seq,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
    )
    event.update({
        "candidate_id": candidate["candidate_id"],
        "source_schema_version": candidate["source_schema_version"],
        "source_t": candidate["source_t"],
        "source_line": candidate["source_line"],
        "source_record_sha256": candidate["source_record_sha256"],
        "config_version_id": candidate["config_version_id"],
        "instrument_id": candidate["instrument_id"],
        "venue_id": candidate["venue_id"],
        "side": candidate["side"],
        "layer": candidate["layer"],
        "passive_price": candidate["price"],
        "candidate_size_Q": candidate["size"],
        "candidate_notional_usd": candidate["candidate_notional_usd"],
        "selected_size_Q": candidate["size"],
        "local_edge_feature": candidate["local_edge_feature"],
        "pair_edge_feature": candidate["pair_edge_feature"],
        "quote_state": candidate["quote_state"],
        "suppression_reason": candidate["suppression_reason"],
        "fair_value": candidate["fair_value"],
        "book_age_ms": candidate["book_age_ms"],
        "book_stale_threshold_ms": candidate["book_stale_threshold_ms"],
        "distance_to_touch_bps": candidate["distance_to_touch_bps"],
        "edge_threshold": candidate["edge_threshold"],
        "candidate_edge_pre_utility": candidate["candidate_edge_pre_utility"],
        "size_raw": candidate["size_raw"],
        "size_margin_cap": candidate["size_margin_cap"],
        "utility_tier": candidate["utility_tier"],
        "utility_role": candidate["utility_role"],
        "utility_reason": candidate["utility_reason"],
        "model_features_hash": candidate["model_features_hash"],
        "calibration_bucket_id": candidate["calibration_bucket_id"],
        "calibration_status": candidate["calibration_status"],
        "calibration_sample_count": candidate["calibration_sample_count"],
        "pair_conditioned_flag": candidate["pair_conditioned_flag"],
        "fast_hedge_allowed": candidate["fast_hedge_allowed"],
        "fast_hedge_serialization_state": candidate["fast_hedge_serialization_state"],
        "residual_state_required": candidate["residual_state_required"],
        "residual_state_status": candidate["residual_state_status"],
        "action_owner": candidate["action_owner"],
        "double_action_prevention_state": candidate["double_action_prevention_state"],
        "min_quote_candidates_required": candidate["min_quote_candidates_required"],
        "min_fill_labels_required": candidate["min_fill_labels_required"],
        "min_hedge_labels_required": candidate["min_hedge_labels_required"],
        "binding_constraints": candidate["binding_constraints"],
        "P_fill_hat": 0.0,
        "P_fill_ci_low": 0.0,
        "P_fill_ci_high": 0.0,
        "P_hedge_success_hat": 0.0,
        "P_hedge_partial_hat": 0.0,
        "P_hedge_fail_hat": 1.0,
        "E_locked_edge_hat": 0.0,
        "E_partial_hedge_state_hat": 0.0,
        "E_residual_inventory_state_hat": 0.0,
        "E_adverse_selection_hat": 0.0,
        "E_queue_reset_hat": 0.0,
        "E_churn_hat": 0.0,
        "E_capital_funding_hat": 0.0,
        "E_tail_risk_hat": 0.0,
        "EV_hat": 0.0,
        "EV_standard_error": 0.0,
        "EV_lcb_alpha": 0.0,
        "alpha": alpha,
        "decision": "HOLD",
        "decision_reason_primary": candidate["decision_reason_primary"],
        "decision_reason_secondary_list": candidate["decision_reason_secondary_list"],
        "admissible_for_financial_claim": False,
    })
    return event


def _replay_label_event(
    candidate: dict[str, Any],
    *,
    ev_event_seq: int,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
) -> dict[str, Any]:
    replay_key_payload = {
        "candidate_id": candidate["candidate_id"],
        "ev_event_seq": ev_event_seq,
        "run_id": run_id,
        "source_record_sha256": candidate["source_record_sha256"],
    }
    event = _base_event(
        event_type="V2_REPLAY_LABEL",
        event_seq=event_seq,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
    )
    event.update({
        "candidate_id": candidate["candidate_id"],
        "source_schema_version": candidate["source_schema_version"],
        "source_t": candidate["source_t"],
        "source_line": candidate["source_line"],
        "source_record_sha256": candidate["source_record_sha256"],
        "venue_id": candidate["venue_id"],
        "side": candidate["side"],
        "layer": candidate["layer"],
        "label_type": "COUNTERFACTUAL_DECISION",
        "source_event_ids": [f"v1:{candidate['source_t']}:{candidate['source_line']}"],
        "deterministic_replay_key": _stable_hash(replay_key_payload),
        "simulator_version": "phase51_ev_shadow_v1",
        "assumptions_hash": _stable_hash({
            "baseline_commit": BASELINE_COMMIT,
            "mode": "nonlive_shadow_hold",
            "reason": "phase51_calibration_hold",
            "secondary_reasons": CALIBRATION_HOLD_REASONS,
        }),
        "label_confidence": 1.0,
        "decision": "HOLD",
        "decision_reason_primary": "counterfactual_decision_not_executed",
        "decision_reason_secondary_list": candidate["decision_reason_secondary_list"],
        "admissible_for_financial_claim": False,
    })
    return event


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(artifact_paths)
    ]


def _write_evidence_index(out_dir: Path, artifact_paths: list[Path], metadata: dict[str, Any]) -> Path:
    evidence_dir = out_dir / "evidence_pack"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "metadata": metadata,
        "artifacts": _artifact_infos(out_dir, artifact_paths),
    }
    index_path = evidence_dir / "artifact_index.json"
    _write_json(index_path, index)
    return index_path


def _write_root_manifest(
    out_dir: Path,
    artifact_paths: list[Path],
    metadata: dict[str, Any],
    created_utc: str,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_utc": created_utc,
        "created_utc_semantics": "deterministic_replay_timestamp_not_wall_clock",
        "metadata": metadata,
        "files": _artifact_infos(out_dir, artifact_paths),
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def run(
    spec_path: Path,
    output_root: Path | None,
    input_telemetry: Path | None,
    run_id: str | None,
    replay_timestamp_ns: int | None = None,
    input_artifact_mode: str = "copy",
) -> Path:
    spec = _load_json(spec_path)
    _validate_spec(spec)
    run_id = run_id or f"{spec.get('experiment_id', 'phase51_ev_shadow')}_{_utc_stamp()}"
    out_dir = _resolve_output_dir(spec, output_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(spec.get("alpha", 0.05))
    input_path = input_telemetry or (
        Path(spec["input_telemetry"]) if spec.get("input_telemetry") else None
    )
    if input_path and not input_path.is_absolute():
        input_path = ROOT / input_path
    if input_artifact_mode not in {"copy", "reference"}:
        raise ValueError("input_artifact_mode must be copy or reference")
    input_sha256 = _sha256(input_path) if input_path else None
    timestamp_ns = replay_timestamp_ns or _derive_replay_timestamp_ns(spec, run_id, input_sha256)
    replay_created_utc = _timestamp_ns_to_utc(timestamp_ns)

    resolved_spec = dict(spec)
    resolved_spec["run_id"] = run_id
    resolved_spec["spec_path"] = str(spec_path)
    resolved_spec["input_telemetry"] = str(input_path) if input_path else spec.get("input_telemetry")
    resolved_spec["input_sha256"] = input_sha256
    resolved_spec["input_artifact_mode"] = input_artifact_mode
    resolved_spec["output_dir"] = str(out_dir)
    resolved_spec["replay_timestamp_ns"] = timestamp_ns
    resolved_spec["replay_created_utc"] = replay_created_utc

    events: list[dict[str, Any]] = [
        {
            **_base_event(
                event_type="V2_RUN_CONTEXT",
                event_seq=1,
                run_id=run_id,
                timestamp_ns=timestamp_ns,
            ),
            "venue_id": "lighter",
            "decision": "HOLD",
            "decision_reason_primary": "nonlive_ev_shadow_context",
            "admissible_for_financial_claim": False,
        }
    ]

    scanned_records = 0
    candidates = 0
    replay_labels = 0
    hold_reason_counts = {reason: 0 for reason in CALIBRATION_HOLD_REASONS}
    if input_path:
        scanned_records, records = _iter_input_records(input_path)
        for line_num, record_hash, record in records:
            for candidate in _quote_candidates_from_record(
                record,
                line_num=line_num,
                record_hash=record_hash,
                run_id=run_id,
                venue_id="lighter",
                instrument_id=str(spec.get("instrument_id", "ETH-PERP")),
            ):
                candidates += 1
                ev_event_seq = len(events) + 1
                events.append(_ev_event(
                    candidate,
                    event_seq=ev_event_seq,
                    run_id=run_id,
                    timestamp_ns=timestamp_ns,
                    alpha=alpha,
                ))
                replay_labels += 1
                events.append(_replay_label_event(
                    candidate,
                    ev_event_seq=ev_event_seq,
                    event_seq=len(events) + 1,
                    run_id=run_id,
                    timestamp_ns=timestamp_ns,
                ))
                for reason in candidate["decision_reason_secondary_list"]:
                    hold_reason_counts[reason] = hold_reason_counts.get(reason, 0) + 1
    else:
        events.append({
            **_base_event(
                event_type="V2_GUARDRAIL_EVENT",
                event_seq=2,
                run_id=run_id,
                timestamp_ns=timestamp_ns,
            ),
            "venue_id": "lighter",
            "decision": "HOLD",
            "decision_reason_primary": "no_input_telemetry",
            "admissible_for_financial_claim": False,
        })

    summary = {
        "experiment_id": spec.get("experiment_id"),
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "venue_id": "lighter",
        "input_records_scanned": scanned_records,
        "input_sha256": input_sha256,
        "candidates_evaluated": candidates,
        "replay_labels_emitted": replay_labels,
        "replay_timestamp_ns": timestamp_ns,
        "replay_created_utc": replay_created_utc,
        "admit_count": 0,
        "reject_count": 0,
        "hold_count": candidates + (1 if not input_path else 0),
        "hold_reason_counts": hold_reason_counts,
        "calibration_status": CALIBRATION_STATUS,
        "min_quote_candidates_required": MIN_QUOTE_CANDIDATES_REQUIRED,
        "min_fill_labels_required": MIN_FILL_LABELS_REQUIRED,
        "min_hedge_labels_required": MIN_HEDGE_LABELS_REQUIRED,
        "gate_status": "HOLD",
        "gate_reason": "nonlive_shadow_requires_calibration_and_board_review",
        "no_live_flag": True,
        "admissible_for_financial_claim": False,
    }
    gate = {
        "status": "HOLD",
        "reason": summary["gate_reason"],
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_nonlive_evidence_review": candidates > 0,
    }
    command_log = {
        "argv": sys.argv,
        "created_utc": replay_created_utc,
        "created_utc_semantics": "deterministic_replay_timestamp_not_wall_clock",
        "python_version": sys.version.split()[0],
        "replay_timestamp_ns": timestamp_ns,
        "input_telemetry": str(input_path) if input_path else None,
        "input_sha256": input_sha256,
        "input_artifact_mode": input_artifact_mode,
    }

    artifact_paths = [
        out_dir / "spec_resolved.json",
        out_dir / "telemetry.jsonl",
        out_dir / "ev_shadow_summary.json",
        out_dir / "gate_result.json",
        out_dir / "command_log.json",
    ]
    _write_json(artifact_paths[0], resolved_spec)
    _write_jsonl(artifact_paths[1], events)
    _write_json(artifact_paths[2], summary)
    _write_json(artifact_paths[3], gate)
    _write_json(artifact_paths[4], command_log)
    if input_path and input_artifact_mode == "copy":
        copied = out_dir / "input_telemetry.source.jsonl"
        shutil.copyfile(input_path, copied)
        artifact_paths.append(copied)

    manifest_metadata = {
        "experiment_id": spec.get("experiment_id"),
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "capital_escalation_flag": False,
        "risk_limit_override_flag": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "input_sha256": input_sha256,
        "input_telemetry": str(input_path) if input_path else None,
        "input_artifact_mode": input_artifact_mode,
        "replay_timestamp_ns": timestamp_ns,
        "replay_created_utc": replay_created_utc,
    }
    evidence_index = _write_evidence_index(out_dir, artifact_paths, manifest_metadata)
    artifact_paths.append(evidence_index)
    _write_root_manifest(out_dir, artifact_paths, manifest_metadata, replay_created_utc)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs/phase51_lighter_only_ev_shadow.json",
        help="Phase 5.1 EV shadow experiment spec JSON.",
    )
    parser.add_argument("--input-telemetry", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--replay-timestamp-ns",
        type=int,
        default=None,
        help="Optional deterministic base timestamp for replay events.",
    )
    parser.add_argument(
        "--input-artifact-mode",
        choices=("copy", "reference"),
        default="copy",
        help="Use 'reference' for large immutable inputs: record path and SHA without copying the source JSONL.",
    )
    args = parser.parse_args()

    try:
        out_dir = run(
            args.spec,
            args.output_root,
            args.input_telemetry,
            args.run_id,
            replay_timestamp_ns=args.replay_timestamp_ns,
            input_artifact_mode=args.input_artifact_mode,
        )
    except Exception as exc:
        print(f"phase51_ev_shadow: ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"phase51_ev_shadow: wrote {out_dir}")
    print("phase51_ev_shadow: status HOLD (nonlive evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
