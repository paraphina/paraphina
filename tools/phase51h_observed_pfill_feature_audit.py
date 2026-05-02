#!/usr/bin/env python3
"""Build the Phase 5.1h observed-only P_fill feature-readiness audit pack.

This is an offline evidence gate. It consumes the Phase 5.1g observed-only
P_fill pack and reconciles available queue/churn, native-limit, and markout
readiness context. It does not train a model, submit orders, approve EV
admission, approve live/canary use, or make financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51h_observed_pfill_feature_audit"

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


def _load_hold_summary(path: Path, filename: str) -> dict[str, Any]:
    summary_path = path / filename
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    return summary


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"bid", "buy"}:
        return "BID"
    if side in {"ask", "sell"}:
        return "ASK"
    return "UNKNOWN"


def _count_map(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _sum_int(records: list[dict[str, Any]], field: str) -> int:
    total = 0
    for record in records:
        value = _safe_int(record.get(field))
        if value is not None:
            total += value
    return total


def _load_pfill_labels(observed_pfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_hold_summary(observed_pfill_run, "pfill_outcome_summary.json")
    if int(summary.get("censored_count") or 0) != 0:
        raise ValueError(f"{observed_pfill_run} must be an observed-only pack with censored_count=0")
    labels_path = observed_pfill_run / "pfill_order_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        outcome_status = str(label.get("outcome_status") or "")
        if outcome_status not in {"OBSERVED_FILLED", "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL"}:
            raise ValueError(f"{labels_path} contains non-observed outcome_status={outcome_status}")
        if outcome_status == "OBSERVED_FILLED" and label.get("p_fill_outcome") != 1.0:
            raise ValueError("OBSERVED_FILLED labels must carry p_fill_outcome=1.0")
        if outcome_status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL" and label.get("p_fill_outcome") != 0.0:
            raise ValueError("OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL labels must carry p_fill_outcome=0.0")
        labels.append(label)
    expected = int(summary.get("order_label_count") or 0)
    if len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary order_label_count {expected}")
    return summary, labels


def _load_canonical_source_index(canonical_pfill_run: Path) -> dict[str, list[str]]:
    manifest_path = canonical_pfill_run / "source_to_canonical_order_manifest.jsonl"
    index: dict[str, list[str]] = {}
    source_to_canonical: dict[str, str] = {}
    for _, row in _iter_jsonl(manifest_path):
        canonical_group_id = str(row.get("canonical_group_id") or "")
        source_order_key = str(row.get("source_order_key") or "")
        if not canonical_group_id or not source_order_key:
            raise ValueError(f"{manifest_path} row missing canonical_group_id/source_order_key")
        previous = source_to_canonical.get(source_order_key)
        if previous is not None and previous != canonical_group_id:
            raise ValueError(f"{manifest_path} source_order_key maps to multiple canonical groups")
        source_to_canonical[source_order_key] = canonical_group_id
        index.setdefault(canonical_group_id, []).append(source_order_key)
    return index


def _load_quarantine_reconciliation(quarantine_review_run: Path) -> dict[str, dict[str, Any]]:
    _load_hold_summary(quarantine_review_run, "quarantine_review_summary.json")
    manifest_path = quarantine_review_run / "source_reconciliation_manifest.jsonl"
    index: dict[str, dict[str, Any]] = {}
    for _, row in _iter_jsonl(manifest_path):
        canonical_group_id = str(row.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError(f"{manifest_path} row missing canonical_group_id")
        if canonical_group_id in index:
            raise ValueError(f"{manifest_path} duplicate canonical_group_id={canonical_group_id}")
        index[canonical_group_id] = row
    return index


def _load_queue_churn(queue_churn_runs: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_order_key: dict[str, list[dict[str, Any]]] = {}
    run_summaries: list[dict[str, Any]] = []
    for run in queue_churn_runs:
        summary = _load_hold_summary(run, "queue_churn_summary.json")
        run_summaries.append({
            "run_id": summary.get("run_id"),
            "run_path": str(run),
            "queue_churn_summary_sha256": _sha256_file(run / "queue_churn_summary.json"),
            "queue_churn_labels_sha256": _sha256_file(run / "queue_churn_labels.jsonl"),
            "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
            "gate_reason": summary.get("gate_reason"),
            "gate_status": summary.get("gate_status"),
        })
        labels_path = run / "queue_churn_labels.jsonl"
        for _, label in _iter_jsonl(labels_path):
            if label.get("label_type") != "QUEUE_CHURN_LABEL":
                continue
            _check_unsafe(label, labels_path, label="label")
            order_key = str(label.get("order_key") or "")
            if not order_key:
                raise ValueError(f"{labels_path} row missing order_key")
            if order_key in by_order_key:
                raise ValueError(f"{labels_path} duplicate queue/churn row for order_key={order_key}")
            by_order_key.setdefault(order_key, []).append(label)
    return by_order_key, run_summaries


def _load_markout_readiness(markout_readiness_runs: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, dict[str, Any]] = {}
    run_summaries: list[dict[str, Any]] = []
    for run in markout_readiness_runs:
        summary = _load_hold_summary(run, "markout_calibration_readiness_summary.json")
        info = {
            "run_id": summary.get("run_id"),
            "run_path": str(run),
            "markout_summary_sha256": _sha256_file(run / "markout_calibration_readiness_summary.json"),
            "gate_reason": summary.get("gate_reason"),
            "gate_status": summary.get("gate_status"),
            "source_telemetry_sha256_list": summary.get("source_telemetry_sha256_list") or [],
        }
        run_summaries.append(info)
        for source_sha in info["source_telemetry_sha256_list"]:
            by_source[str(source_sha)] = info
    return by_source, run_summaries


def _source_order_keys(label: dict[str, Any], canonical_index: dict[str, list[str]]) -> list[str]:
    keys = [str(value) for value in (label.get("source_order_keys") or []) if value]
    canonical_group_id = str(label.get("canonical_group_id") or "")
    indexed = list(canonical_index.get(canonical_group_id) or [])
    source_label_count = _safe_int(label.get("source_label_count"))
    if source_label_count is not None and indexed and len(indexed) != source_label_count:
        raise ValueError(
            f"source_to_canonical row count {len(indexed)} != source_label_count {source_label_count} "
            f"for canonical_group_id={canonical_group_id}"
        )
    if keys and indexed and set(keys) != set(indexed):
        raise ValueError(f"source_order_keys mismatch for canonical_group_id={canonical_group_id}")
    if not keys:
        keys = indexed
    if not keys and label.get("order_key"):
        keys = [str(label["order_key"])]
    return sorted(set(keys))


def _native_status_class(venue: str, queue_labels: list[dict[str, Any]]) -> str:
    if venue != "lighter":
        return "NOT_APPLICABLE_NON_LIGHTER"
    statuses = {str(label.get("native_limit_pressure_status") or "UNKNOWN") for label in queue_labels}
    if any(status.startswith("OBSERVED") for status in statuses):
        return "OBSERVED"
    if any(status.startswith("PARTIAL") for status in statuses):
        return "PARTIAL"
    return "UNKNOWN"


def _maker_taker_status(label: dict[str, Any]) -> str:
    fill_count = _safe_int(label.get("fill_count")) or 0
    if fill_count <= 0:
        return "NO_FILL_NOT_APPLICABLE"
    counts = label.get("maker_taker_role_counts") or {}
    known = int(counts.get("MAKER") or 0) + int(counts.get("TAKER") or 0)
    unknown = int(counts.get("UNKNOWN") or 0)
    if known >= fill_count and unknown == 0:
        return "OBSERVED"
    if unknown > 0 or known > 0:
        return "PARTIAL_OR_UNKNOWN"
    return "MISSING"


def _coverage_label(
    *,
    seq: int,
    run_id: str,
    timestamp_ns: int,
    pfill_label: dict[str, Any],
    source_order_keys: list[str],
    queue_labels: list[dict[str, Any]],
    quarantine_record: dict[str, Any] | None,
    markout_info: dict[str, Any] | None,
) -> dict[str, Any]:
    venue = str(pfill_label.get("venue_id") or "UNKNOWN")
    side = _canonical_side(pfill_label.get("side"))
    source_key_count = len(source_order_keys)
    matched_source_keys = {
        str(label.get("order_key"))
        for label in queue_labels
        if label.get("order_key")
    }
    if not queue_labels:
        queue_status = "MISSING"
    elif len(matched_source_keys) >= source_key_count:
        queue_status = "JOINED_ALL_SOURCE_KEYS"
    else:
        queue_status = "JOINED_PARTIAL_SOURCE_KEYS"
    native_status = _native_status_class(venue, queue_labels)
    maker_taker_status = _maker_taker_status(pfill_label)
    observed_horizon_available = pfill_label.get("observed_horizon_source_ticks") is not None
    markout_available = markout_info is not None
    raw_decision_id_present = pfill_label.get("decision_id") not in (None, "")
    missing_features: list[str] = []
    if not observed_horizon_available:
        missing_features.append("missing_observed_horizon")
    if queue_status == "MISSING":
        missing_features.append("missing_queue_churn_join")
    elif queue_status == "JOINED_PARTIAL_SOURCE_KEYS":
        missing_features.append("partial_queue_churn_join")
    if venue == "lighter" and native_status != "OBSERVED":
        missing_features.append("lighter_native_limit_pressure_not_observed")
    if maker_taker_status in {"PARTIAL_OR_UNKNOWN", "MISSING"}:
        missing_features.append("maker_taker_not_fully_observed_for_filled_order")
    if not markout_available:
        missing_features.append("missing_markout_readiness_source_context")
    if raw_decision_id_present:
        missing_features.append("raw_decision_id_present_in_input_not_emitted")

    return {
        "schema_version": 1,
        "label_type": "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL",
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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
        "canonical_group_id": pfill_label.get("canonical_group_id"),
        "canonical_order_key": pfill_label.get("order_key"),
        "source_telemetry_sha256": pfill_label.get("source_telemetry_sha256"),
        "venue_id": venue,
        "side": side,
        "order_holdout_split": str(pfill_label.get("order_holdout_split") or "UNKNOWN").upper(),
        "outcome_status": pfill_label.get("outcome_status"),
        "p_fill_outcome": pfill_label.get("p_fill_outcome"),
        "fill_count": _safe_int(pfill_label.get("fill_count")) or 0,
        "terminal_action_first": pfill_label.get("terminal_action_first"),
        "terminal_event_count": _safe_int(pfill_label.get("terminal_event_count")) or 0,
        "observed_horizon_available": observed_horizon_available,
        "observed_horizon_source_ticks": pfill_label.get("observed_horizon_source_ticks"),
        "source_order_key_count": source_key_count,
        "source_queue_churn_label_count": len(queue_labels),
        "queue_churn_join_status": queue_status,
        "queue_lifecycle_join_status_counts": _count_map(queue_labels, "lifecycle_join_status"),
        "queue_reset_proxy_event_count": _sum_int(queue_labels, "queue_reset_proxy_event_count"),
        "replace_event_count": _sum_int(queue_labels, "replace_event_count"),
        "cancel_event_count": _sum_int(queue_labels, "cancel_event_count"),
        "cancel_all_event_count": _sum_int(queue_labels, "cancel_all_event_count"),
        "churn_event_count": _sum_int(queue_labels, "churn_event_count"),
        "native_limit_pressure_status": native_status,
        "native_limit_pressure_status_counts": _count_map(queue_labels, "native_limit_pressure_status"),
        "maker_taker_feature_status": maker_taker_status,
        "maker_taker_role_counts": pfill_label.get("maker_taker_role_counts") or {},
        "markout_readiness_source_available": markout_available,
        "markout_readiness_gate_reason": markout_info.get("gate_reason") if markout_info else None,
        "quarantine_review_status": quarantine_record.get("review_status") if quarantine_record else None,
        "quarantine_included_in_observed_only_pack": (
            quarantine_record.get("included_in_observed_only_pack") if quarantine_record else None
        ),
        "raw_identifier_input_present": raw_decision_id_present,
        "raw_identifier_redaction_status": (
            "RAW_DECISION_ID_PRESENT_IN_INPUT_NOT_EMITTED"
            if raw_decision_id_present
            else "NO_RAW_DECISION_ID_INPUT"
        ),
        "missing_feature_count": len(missing_features),
        "missing_features": missing_features,
    }


def _empty_bucket_counts() -> dict[str, int]:
    return {
        "label_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "train_count": 0,
        "holdout_count": 0,
        "observed_horizon_available_count": 0,
        "observed_horizon_missing_count": 0,
        "queue_churn_joined_all_count": 0,
        "queue_churn_joined_partial_count": 0,
        "queue_churn_missing_count": 0,
        "queue_reset_proxy_present_count": 0,
        "native_limit_observed_count": 0,
        "native_limit_partial_count": 0,
        "native_limit_unknown_count": 0,
        "native_limit_not_applicable_count": 0,
        "maker_taker_observed_count": 0,
        "maker_taker_partial_or_unknown_count": 0,
        "maker_taker_missing_count": 0,
        "maker_taker_not_applicable_count": 0,
        "markout_source_available_count": 0,
        "markout_source_missing_count": 0,
        "raw_identifier_input_present_count": 0,
        "missing_feature_total": 0,
    }


def _add_to_bucket(counts: dict[str, int], label: dict[str, Any]) -> None:
    counts["label_count"] += 1
    if label.get("p_fill_outcome") == 1.0:
        counts["filled_count"] += 1
    else:
        counts["not_filled_count"] += 1
    split = str(label.get("order_holdout_split") or "UNKNOWN")
    if split == "HOLDOUT":
        counts["holdout_count"] += 1
    else:
        counts["train_count"] += 1
    if label.get("observed_horizon_available"):
        counts["observed_horizon_available_count"] += 1
    else:
        counts["observed_horizon_missing_count"] += 1
    queue_status = label.get("queue_churn_join_status")
    if queue_status == "JOINED_ALL_SOURCE_KEYS":
        counts["queue_churn_joined_all_count"] += 1
    elif queue_status == "JOINED_PARTIAL_SOURCE_KEYS":
        counts["queue_churn_joined_partial_count"] += 1
    else:
        counts["queue_churn_missing_count"] += 1
    if int(label.get("queue_reset_proxy_event_count") or 0) > 0:
        counts["queue_reset_proxy_present_count"] += 1
    native_status = label.get("native_limit_pressure_status")
    if native_status == "OBSERVED":
        counts["native_limit_observed_count"] += 1
    elif native_status == "PARTIAL":
        counts["native_limit_partial_count"] += 1
    elif native_status == "UNKNOWN":
        counts["native_limit_unknown_count"] += 1
    else:
        counts["native_limit_not_applicable_count"] += 1
    maker_status = label.get("maker_taker_feature_status")
    if maker_status == "OBSERVED":
        counts["maker_taker_observed_count"] += 1
    elif maker_status == "PARTIAL_OR_UNKNOWN":
        counts["maker_taker_partial_or_unknown_count"] += 1
    elif maker_status == "MISSING":
        counts["maker_taker_missing_count"] += 1
    else:
        counts["maker_taker_not_applicable_count"] += 1
    if label.get("markout_readiness_source_available"):
        counts["markout_source_available_count"] += 1
    else:
        counts["markout_source_missing_count"] += 1
    if label.get("raw_identifier_input_present"):
        counts["raw_identifier_input_present_count"] += 1
    counts["missing_feature_total"] += int(label.get("missing_feature_count") or 0)


def _bucket_entries(label: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    venue = str(label.get("venue_id") or "UNKNOWN")
    side = str(label.get("side") or "UNKNOWN")
    return [
        ("GLOBAL", {"scope": "GLOBAL"}),
        (f"VENUE:{venue}", {"venue_id": venue}),
        (f"SIDE:{side}", {"side": side}),
        (f"VENUE_SIDE:{venue}:{side}", {"venue_id": venue, "side": side}),
    ]


def _bucket_gate_reasons(
    counts: dict[str, int],
    *,
    dimensions: dict[str, str],
    min_observed_per_bucket: int,
    min_holdout_observed_per_bucket: int,
) -> list[str]:
    reasons: list[str] = []
    if (
        counts["label_count"] < min_observed_per_bucket
        or counts["holdout_count"] < min_holdout_observed_per_bucket
    ):
        reasons.append("sparse_pfill_feature_bucket")
    if counts["observed_horizon_missing_count"] > 0:
        reasons.append("missing_observed_horizon_features")
    if counts["queue_churn_missing_count"] > 0 or counts["queue_churn_joined_partial_count"] > 0:
        reasons.append("queue_churn_join_incomplete")
    if dimensions.get("venue_id") == "lighter" and (
        counts["native_limit_partial_count"] > 0 or counts["native_limit_unknown_count"] > 0
    ):
        reasons.append("lighter_native_limit_pressure_not_fully_observed")
    if counts["maker_taker_partial_or_unknown_count"] > 0 or counts["maker_taker_missing_count"] > 0:
        reasons.append("maker_taker_not_fully_observed_for_filled_orders")
    if counts["markout_source_missing_count"] > 0:
        reasons.append("markout_readiness_source_context_missing")
    if counts["raw_identifier_input_present_count"] > 0:
        reasons.append("raw_identifier_present_in_input_not_emitted")
    reasons.append("requires_feature_rich_pfill_model_and_board_review")
    return reasons


def _build_bucket_records(
    labels: list[dict[str, Any]],
    *,
    run_id: str,
    timestamp_ns: int,
    min_observed_per_bucket: int,
    min_holdout_observed_per_bucket: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for label in labels:
        for bucket_id, dimensions in _bucket_entries(label):
            bucket = buckets.setdefault(bucket_id, {"dimensions": dimensions, "counts": _empty_bucket_counts()})
            _add_to_bucket(bucket["counts"], label)
    records: list[dict[str, Any]] = []
    for seq, (bucket_id, bucket) in enumerate(sorted(buckets.items()), start=1):
        counts = bucket["counts"]
        dimensions = bucket["dimensions"]
        records.append({
            "schema_version": 1,
            "label_type": "PHASE51H_PFILL_FEATURE_BUCKET_READINESS",
            "bucket_seq": seq,
            "timestamp_local_ns": timestamp_ns + seq,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "bucket_id": bucket_id,
            "bucket_dimensions": dimensions,
            "gate_status": "HOLD",
            "gate_reasons": _bucket_gate_reasons(
                counts,
                dimensions=dimensions,
                min_observed_per_bucket=min_observed_per_bucket,
                min_holdout_observed_per_bucket=min_holdout_observed_per_bucket,
            ),
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
            "min_observed_per_bucket": min_observed_per_bucket,
            "min_holdout_observed_per_bucket": min_holdout_observed_per_bucket,
            **counts,
        })
    return records


def _summary_gate_reason(bucket_records: list[dict[str, Any]]) -> str:
    all_reasons = {
        reason
        for bucket in bucket_records
        for reason in bucket.get("gate_reasons", [])
    }
    priority = [
        "raw_identifier_present_in_input_not_emitted",
        "missing_observed_horizon_features",
        "queue_churn_join_incomplete",
        "lighter_native_limit_pressure_not_fully_observed",
        "maker_taker_not_fully_observed_for_filled_orders",
        "markout_readiness_source_context_missing",
        "sparse_pfill_feature_bucket",
    ]
    for reason in priority:
        if reason in all_reasons:
            return f"phase51h_{reason}"
    return "phase51h_requires_feature_rich_pfill_model_and_board_review"


def build_feature_audit(
    *,
    observed_pfill_run: Path,
    quarantine_review_run: Path,
    canonical_pfill_run: Path,
    queue_churn_runs: list[Path],
    markout_readiness_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    min_observed_per_bucket: int,
    min_holdout_observed_per_bucket: int,
) -> Path:
    run_id = run_id or f"PHASE51H-OBSERVED-PFILL-FEATURE-AUDIT-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    observed_pfill_run = _resolve_path(observed_pfill_run)
    quarantine_review_run = _resolve_path(quarantine_review_run)
    canonical_pfill_run = _resolve_path(canonical_pfill_run)
    queue_churn_runs = [_resolve_path(run) for run in queue_churn_runs]
    markout_readiness_runs = [_resolve_path(run) for run in markout_readiness_runs]

    pfill_summary, pfill_labels = _load_pfill_labels(observed_pfill_run)
    canonical_index = _load_canonical_source_index(canonical_pfill_run)
    quarantine_index = _load_quarantine_reconciliation(quarantine_review_run)
    queue_by_order_key, queue_run_summaries = _load_queue_churn(queue_churn_runs)
    markout_by_source, markout_run_summaries = _load_markout_readiness(markout_readiness_runs)

    feature_labels: list[dict[str, Any]] = []
    for seq, pfill_label in enumerate(pfill_labels, start=1):
        canonical_group_id = str(pfill_label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError("P_fill label missing canonical_group_id")
        quarantine_record = quarantine_index.get(canonical_group_id)
        if quarantine_record is None:
            raise ValueError(f"canonical_group_id {canonical_group_id} missing from quarantine reconciliation")
        if quarantine_record.get("included_in_observed_only_pack") is not True:
            raise ValueError(f"canonical_group_id {canonical_group_id} is not marked included in observed-only pack")
        if canonical_group_id not in canonical_index:
            raise ValueError(f"canonical_group_id {canonical_group_id} missing from source-to-canonical manifest")
        source_keys = _source_order_keys(pfill_label, canonical_index)
        missing_queue_keys = [source_key for source_key in source_keys if source_key not in queue_by_order_key]
        if missing_queue_keys:
            raise ValueError(
                f"canonical_group_id {canonical_group_id} missing queue/churn rows for "
                f"{len(missing_queue_keys)} source order key(s)"
            )
        queue_labels = [
            label
            for source_key in source_keys
            for label in queue_by_order_key.get(source_key, [])
        ]
        source_sha = str(pfill_label.get("source_telemetry_sha256") or "")
        for queue_label in queue_labels:
            if str(queue_label.get("source_telemetry_sha256") or "") != source_sha:
                raise ValueError(f"queue/churn source hash mismatch for canonical_group_id={canonical_group_id}")
        markout_info = markout_by_source.get(str(pfill_label.get("source_telemetry_sha256") or ""))
        feature_labels.append(_coverage_label(
            seq=seq,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            pfill_label=pfill_label,
            source_order_keys=source_keys,
            queue_labels=queue_labels,
            quarantine_record=quarantine_record,
            markout_info=markout_info,
        ))

    bucket_records = _build_bucket_records(
        feature_labels,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        min_observed_per_bucket=min_observed_per_bucket,
        min_holdout_observed_per_bucket=min_holdout_observed_per_bucket,
    )
    global_bucket = next(record for record in bucket_records if record["bucket_id"] == "GLOBAL")
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": _summary_gate_reason(bucket_records),
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
        "observed_pfill_run": str(observed_pfill_run),
        "quarantine_review_run": str(quarantine_review_run),
        "canonical_pfill_run": str(canonical_pfill_run),
        "observed_pfill_summary_sha256": _sha256_file(observed_pfill_run / "pfill_outcome_summary.json"),
        "observed_pfill_labels_sha256": _sha256_file(observed_pfill_run / "pfill_order_labels.jsonl"),
        "quarantine_reconciliation_sha256": _sha256_file(quarantine_review_run / "source_reconciliation_manifest.jsonl"),
        "canonical_source_manifest_sha256": _sha256_file(canonical_pfill_run / "source_to_canonical_order_manifest.jsonl"),
        "queue_churn_runs": queue_run_summaries,
        "markout_readiness_runs": markout_run_summaries,
        "source_telemetry_sha256_list": sorted({
            str(label.get("source_telemetry_sha256") or "")
            for label in pfill_labels
            if label.get("source_telemetry_sha256")
        }),
        "input_pfill_gate_reason": pfill_summary.get("gate_reason"),
        "excluded_quarantine_count": int(pfill_summary.get("excluded_quarantine_count") or 0),
        "excluded_quarantine_reason_counts": pfill_summary.get("excluded_quarantine_reason_counts") or {},
        "observed_only_pack_warning": pfill_summary.get("observed_only_pack_warning"),
        "bucket_count": len(bucket_records),
        **{
            key: global_bucket[key]
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
    }

    labels_path = out_dir / "pfill_feature_coverage_labels.jsonl"
    buckets_path = out_dir / "pfill_feature_bucket_readiness.jsonl"
    summary_path = out_dir / "pfill_feature_audit_summary.json"
    _write_jsonl(labels_path, feature_labels)
    _write_jsonl(buckets_path, bucket_records)
    _write_json(summary_path, summary)

    artifact_paths = [labels_path, buckets_path, summary_path]
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
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--quarantine-review-run", type=Path, required=True)
    parser.add_argument("--canonical-pfill-run", type=Path, required=True)
    parser.add_argument("--queue-churn-run", type=Path, action="append", default=[])
    parser.add_argument("--markout-readiness-run", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--min-observed-per-bucket", type=int, default=200)
    parser.add_argument("--min-holdout-observed-per-bucket", type=int, default=50)
    args = parser.parse_args()
    try:
        out_dir = build_feature_audit(
            observed_pfill_run=args.observed_pfill_run,
            quarantine_review_run=args.quarantine_review_run,
            canonical_pfill_run=args.canonical_pfill_run,
            queue_churn_runs=args.queue_churn_run,
            markout_readiness_runs=args.markout_readiness_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            min_observed_per_bucket=args.min_observed_per_bucket,
            min_holdout_observed_per_bucket=args.min_holdout_observed_per_bucket,
        )
    except Exception as exc:
        print(f"phase51h_observed_pfill_feature_audit: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51h_observed_pfill_feature_audit: wrote {out_dir}")
    print("phase51h_observed_pfill_feature_audit: status HOLD (feature readiness only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
