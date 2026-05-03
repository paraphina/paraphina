#!/usr/bin/env python3
"""Build the Phase 5.1i P_fill feature-matrix admissibility evidence pack.

This is an offline HOLD-only evidence gate. It consumes a redacted Phase 5.1h
observed P_fill feature audit pack and records whether the current matrix is
admissible for later P_fill calibration. It does not train a model, submit
orders, approve EV admission, approve live/canary use, or make financial claims.
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


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51i_pfill_feature_matrix_admissibility"

UNSAFE_TRUE_FLAGS = {
    "approved_for_model_training",
    "approved_for_live",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_financial_claim",
    "admissible_for_financial_claim",
    "admissible_for_ev_admission",
    "live_orders_allowed",
    "capital_change_allowed",
    "risk_limit_relaxation_allowed",
}

RAW_IDENTIFIER_FIELDS = {
    "decision_id",
    "order_id",
    "client_order_id",
    "venue_order_id",
    "raw_order_id",
    "raw_client_order_id",
}

HORIZON_RECOVERY_SUMMARY_KEYS = (
    "horizon_recovery_run",
    "horizon_recovery_summary_sha256",
    "horizon_recovery_labels_sha256",
    "horizon_recovery_status_counts",
    "horizon_recovery_applied_count",
    "horizon_recovered_terminal_count",
    "horizon_recovery_preserved_existing_count",
    "horizon_recovery_fill_timebase_remaining_count",
    "filled_horizon_recovery_run",
    "filled_horizon_recovery_summary_sha256",
    "filled_horizon_recovery_labels_sha256",
    "filled_horizon_recovery_status_counts",
    "filled_horizon_recovery_applied_count",
    "filled_horizon_recovered_source_tick_count",
    "filled_horizon_exchange_ms_only_count",
    "filled_horizon_unrecovered_count",
    "filled_horizon_source_key_recovery_run",
    "filled_horizon_source_key_recovery_summary_sha256",
    "filled_horizon_source_key_recovery_labels_sha256",
    "filled_horizon_source_key_recovery_status_counts",
    "filled_horizon_source_key_recovery_applied_count",
    "filled_horizon_source_key_recovered_source_tick_count",
    "filled_horizon_source_key_pfill_horizon_recovered_count",
    "filled_horizon_source_key_observed_hash_recovered_count",
    "filled_horizon_source_key_unrecovered_count",
)

HORIZON_RECOVERY_BUCKET_KEYS = (
    "horizon_recovery_applied_count",
    "horizon_recovered_terminal_count",
    "horizon_recovery_preserved_existing_count",
    "horizon_recovery_fill_timebase_remaining_count",
    "filled_horizon_recovery_applied_count",
    "filled_horizon_recovered_source_tick_count",
    "filled_horizon_exchange_ms_only_count",
    "filled_horizon_unrecovered_count",
    "filled_horizon_source_key_recovery_applied_count",
    "filled_horizon_source_key_recovered_source_tick_count",
    "filled_horizon_source_key_pfill_horizon_recovered_count",
    "filled_horizon_source_key_observed_hash_recovered_count",
    "filled_horizon_source_key_unrecovered_count",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = ROOT / path
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
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
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, record


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_no_raw_identifiers(record: dict[str, Any], path: Path, *, label: str) -> None:
    present = sorted(field for field in RAW_IDENTIFIER_FIELDS if field in record)
    if present:
        raise ValueError(f"{path} has raw identifier field(s) in {label}: {', '.join(present)}")


def _load_hold_summary(run_path: Path) -> dict[str, Any]:
    summary_path = run_path / "pfill_feature_audit_summary.json"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    if int(summary.get("raw_identifier_input_present_count") or 0) != 0:
        raise ValueError(f"{summary_path} is not a redacted Phase 5.1h input")
    return summary


def _load_labels(run_path: Path) -> list[dict[str, Any]]:
    labels_path = run_path / "pfill_feature_coverage_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        _check_no_raw_identifiers(label, labels_path, label="label")
        if label.get("raw_identifier_input_present") is True:
            raise ValueError(f"{labels_path} contains unredacted raw identifier input")
        labels.append(label)
    return labels


def _load_buckets(run_path: Path) -> list[dict[str, Any]]:
    buckets_path = run_path / "pfill_feature_bucket_readiness.jsonl"
    buckets: list[dict[str, Any]] = []
    for _, bucket in _iter_jsonl(buckets_path):
        if bucket.get("label_type") != "PHASE51H_PFILL_FEATURE_BUCKET_READINESS":
            continue
        _check_unsafe(bucket, buckets_path, label="bucket")
        _check_no_raw_identifiers(bucket, buckets_path, label="bucket")
        if int(bucket.get("raw_identifier_input_present_count") or 0) != 0:
            raise ValueError(f"{buckets_path} contains unredacted raw identifier input")
        buckets.append(bucket)
    return buckets


def _blocker(
    *,
    seq: int,
    run_id: str,
    timestamp_ns: int,
    blocker_id: str,
    measured_count: int,
    gate_status: str,
    severity: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": "PHASE51I_PFILL_FEATURE_MATRIX_BLOCKER",
        "blocker_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": gate_status,
        "severity": severity,
        "blocker_id": blocker_id,
        "measured_count": measured_count,
        "detail": detail,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "no_live_flag": True,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }


def _build_blockers(summary: dict[str, Any], buckets: list[dict[str, Any]], run_id: str, timestamp_ns: int) -> list[dict[str, Any]]:
    sparse_count = sum(1 for bucket in buckets if "sparse_pfill_feature_bucket" in (bucket.get("gate_reasons") or []))
    specs = [
        (
            "raw_identifier_redaction_passed",
            0,
            "PASS",
            "INFO",
            "no raw decision identifiers are present in the redacted Phase 5.1h input",
        ),
        (
            "filled_horizon_source_key_still_missing",
            int(summary.get("filled_horizon_source_key_unrecovered_count") or 0),
            "HOLD",
            "HARD_BLOCK",
            "filled-order source-tick horizons remain unrecovered after Phase 5.1l source-key recovery",
        ),
        (
            "filled_horizon_source_tick_still_missing",
            int(summary.get("filled_horizon_unrecovered_count") or 0),
            "HOLD",
            "HARD_BLOCK",
            "filled-order source-tick horizons remain unrecovered after Phase 5.1k timebase recovery",
        ),
        (
            "missing_observed_horizon_features",
            int(summary.get("observed_horizon_missing_count") or 0),
            "HOLD",
            "HARD_BLOCK",
            "observed horizon timing is required before P_fill model calibration",
        ),
        (
            "filled_horizon_exchange_ms_only_requires_board_review",
            int(summary.get("filled_horizon_exchange_ms_only_count") or 0),
            "HOLD",
            "HARD_BLOCK",
            "exchange-millisecond filled horizons cannot be used as source-tick horizons without board review",
        ),
        (
            "lighter_native_limit_pressure_not_fully_observed",
            int(summary.get("native_limit_partial_count") or 0) + int(summary.get("native_limit_unknown_count") or 0),
            "HOLD",
            "HOLD",
            "Lighter native-limit pressure is not fully observed for all Lighter rows",
        ),
        (
            "maker_taker_not_fully_observed_for_filled_orders",
            int(summary.get("maker_taker_partial_or_unknown_count") or 0) + int(summary.get("maker_taker_missing_count") or 0),
            "HOLD",
            "HOLD",
            "filled-order maker/taker evidence remains partial or missing",
        ),
        (
            "sparse_pfill_feature_buckets",
            sparse_count,
            "HOLD",
            "HOLD",
            "one or more venue/side feature buckets fail minimum count thresholds",
        ),
        (
            "observed_only_selection_bias_not_resolved",
            int(summary.get("excluded_quarantine_count") or 0),
            "HOLD",
            "HOLD",
            "observed-only diagnostic labels are not unbiased training/admission evidence",
        ),
    ]
    return [
        _blocker(
            seq=seq,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            blocker_id=blocker_id,
            measured_count=measured_count,
            gate_status=gate_status,
            severity=severity,
            detail=detail,
        )
        for seq, (blocker_id, measured_count, gate_status, severity, detail) in enumerate(specs, start=1)
        if gate_status == "PASS" or measured_count > 0
    ]


def _summary_gate_reason(blockers: list[dict[str, Any]]) -> str:
    blocker_ids = {str(blocker.get("blocker_id")) for blocker in blockers if blocker.get("gate_status") == "HOLD"}
    priority = [
        "filled_horizon_source_key_still_missing",
        "filled_horizon_source_tick_still_missing",
        "missing_observed_horizon_features",
        "filled_horizon_exchange_ms_only_requires_board_review",
        "lighter_native_limit_pressure_not_fully_observed",
        "maker_taker_not_fully_observed_for_filled_orders",
        "sparse_pfill_feature_buckets",
        "observed_only_selection_bias_not_resolved",
    ]
    for blocker_id in priority:
        if blocker_id in blocker_ids:
            return f"phase51i_{blocker_id}"
    return "phase51i_feature_matrix_requires_board_review"


def build_matrix_admissibility(
    *,
    feature_audit_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51I-PFILL-FEATURE-MATRIX-ADMISSIBILITY-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    feature_audit_run = _resolve_path(feature_audit_run)
    input_summary = _load_hold_summary(feature_audit_run)
    labels = _load_labels(feature_audit_run)
    buckets = _load_buckets(feature_audit_run)
    expected_label_count = int(input_summary.get("label_count") or 0)
    if len(labels) != expected_label_count:
        raise ValueError(f"feature label count {len(labels)} != summary label_count {expected_label_count}")
    if len(buckets) != int(input_summary.get("bucket_count") or 0):
        raise ValueError("feature bucket count does not reconcile to summary bucket_count")

    blockers = _build_blockers(input_summary, buckets, run_id, timestamp_ns)
    matrix_buckets: list[dict[str, Any]] = []
    for seq, bucket in enumerate(buckets, start=1):
        reasons = list(bucket.get("gate_reasons") or [])
        matrix_buckets.append({
            "schema_version": 1,
            "label_type": "PHASE51I_PFILL_FEATURE_MATRIX_BUCKET",
            "bucket_seq": seq,
            "timestamp_local_ns": timestamp_ns + seq,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "source_phase51h_run": str(feature_audit_run),
            "bucket_id": bucket.get("bucket_id"),
            "bucket_dimensions": bucket.get("bucket_dimensions") or {},
            "matrix_status": "HOLD",
            "matrix_gate_reasons": sorted(set(reasons + ["requires_phase51i_board_review"])),
            "approved_for_model_training": False,
            "approved_for_live": False,
            "approved_for_canary": False,
            "approved_for_capital_escalation": False,
            "admissible_for_financial_claim": False,
            "admissible_for_ev_admission": False,
            "no_live_flag": True,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
            **{
                key: bucket.get(key)
                for key in (
                    "label_count",
                    "filled_count",
                    "not_filled_count",
                    "train_count",
                    "holdout_count",
                    "observed_horizon_available_count",
                    "observed_horizon_missing_count",
                    "queue_churn_joined_all_count",
                    "queue_churn_joined_partial_count",
                    "queue_churn_missing_count",
                    "queue_reset_proxy_present_count",
                    "native_limit_observed_count",
                    "native_limit_partial_count",
                    "native_limit_unknown_count",
                    "native_limit_not_applicable_count",
                    "maker_taker_observed_count",
                    "maker_taker_partial_or_unknown_count",
                    "maker_taker_missing_count",
                    "maker_taker_not_applicable_count",
                    "markout_source_available_count",
                    "markout_source_missing_count",
                    "raw_identifier_input_present_count",
                    "missing_feature_total",
                )
            },
            **{
                key: bucket.get(key)
                for key in HORIZON_RECOVERY_BUCKET_KEYS
                if key in bucket
            },
        })

    blocker_ids = [str(blocker["blocker_id"]) for blocker in blockers if blocker["gate_status"] == "HOLD"]
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": _summary_gate_reason(blockers),
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "no_live_flag": True,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "feature_audit_run": str(feature_audit_run),
        "feature_audit_summary_sha256": _sha256_file(feature_audit_run / "pfill_feature_audit_summary.json"),
        "feature_coverage_labels_sha256": _sha256_file(feature_audit_run / "pfill_feature_coverage_labels.jsonl"),
        "feature_bucket_readiness_sha256": _sha256_file(feature_audit_run / "pfill_feature_bucket_readiness.jsonl"),
        "source_telemetry_sha256_list": input_summary.get("source_telemetry_sha256_list") or [],
        "raw_identifier_redaction_status": "PASS",
        "matrix_admissibility_status": "HOLD",
        "matrix_blocker_ids": blocker_ids,
        "matrix_blocker_count": len(blocker_ids),
        "bucket_count": len(matrix_buckets),
        "excluded_quarantine_count": int(input_summary.get("excluded_quarantine_count") or 0),
        "excluded_quarantine_reason_counts": input_summary.get("excluded_quarantine_reason_counts") or {},
        "observed_only_pack_warning": input_summary.get("observed_only_pack_warning"),
        **{
            key: input_summary.get(key)
            for key in (
                "label_count",
                "filled_count",
                "not_filled_count",
                "train_count",
                "holdout_count",
                "observed_horizon_available_count",
                "observed_horizon_missing_count",
                "queue_churn_joined_all_count",
                "queue_churn_joined_partial_count",
                "queue_churn_missing_count",
                "queue_reset_proxy_present_count",
                "native_limit_observed_count",
                "native_limit_partial_count",
                "native_limit_unknown_count",
                "native_limit_not_applicable_count",
                "maker_taker_observed_count",
                "maker_taker_partial_or_unknown_count",
                "maker_taker_missing_count",
                "maker_taker_not_applicable_count",
                "markout_source_available_count",
                "markout_source_missing_count",
                "raw_identifier_input_present_count",
                "missing_feature_total",
            )
        },
        **{
            key: input_summary.get(key)
            for key in HORIZON_RECOVERY_SUMMARY_KEYS
            if key in input_summary
        },
        "input_summary_hash": _stable_hash(input_summary),
    }

    summary_path = out_dir / "pfill_feature_matrix_admissibility_summary.json"
    buckets_path = out_dir / "pfill_feature_matrix_buckets.jsonl"
    blockers_path = out_dir / "pfill_feature_matrix_blockers.jsonl"
    _write_json(summary_path, summary)
    _write_jsonl(buckets_path, matrix_buckets)
    _write_jsonl(blockers_path, blockers)

    artifact_paths = [summary_path, buckets_path, blockers_path]
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, artifact_paths),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, artifact_paths + [artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-audit-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_matrix_admissibility(
            feature_audit_run=args.feature_audit_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51i_pfill_feature_matrix_admissibility: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51i_pfill_feature_matrix_admissibility: wrote {out_dir}")
    print("phase51i_pfill_feature_matrix_admissibility: status HOLD (matrix admissibility only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
