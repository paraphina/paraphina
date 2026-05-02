#!/usr/bin/env python3
"""Recover Phase 5.1 filled-order horizon timing without mixing timebases.

This is an offline HOLD-only evidence gate. It consumes the redacted Phase 5.1h
feature audit pack, canonical P_fill outcomes, and Phase 5.1e lifecycle truth.
It recovers filled-order source-tick horizons only when both order and fill
source ticks are observable. Exchange-millisecond recovery is recorded as a
separate feature and is never written into observed_horizon_source_ticks.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51k_filled_horizon_timebase_recovery"

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
    "fill_id",
    "trade_id",
    "native_trade_id",
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


def _resolve_path(path: Path | str | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None:
            return parsed
    return None


def _canonical_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"bid", "buy"}:
        return "BID"
    if side in {"ask", "sell"}:
        return "ASK"
    return "UNKNOWN"


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_no_raw_identifiers(record: dict[str, Any], path: Path, *, label: str) -> None:
    present = sorted(field for field in RAW_IDENTIFIER_FIELDS if field in record)
    if present:
        raise ValueError(f"{path} has raw identifier field(s) in {label}: {', '.join(present)}")


def _load_hold_summary(run_path: Path, filename: str) -> dict[str, Any]:
    summary_path = run_path / filename
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    return summary


def _load_feature_audit(feature_audit_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_hold_summary(feature_audit_run, "pfill_feature_audit_summary.json")
    if int(summary.get("raw_identifier_input_present_count") or 0) != 0:
        raise ValueError(f"{feature_audit_run} must be a redacted Phase 5.1h feature audit input")
    labels_path = feature_audit_run / "pfill_feature_coverage_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "PHASE51H_PFILL_FEATURE_COVERAGE_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        _check_no_raw_identifiers(label, labels_path, label="label")
        if label.get("raw_identifier_input_present") is True:
            raise ValueError(f"{labels_path} contains raw identifier input")
        labels.append(label)
    expected = int(summary.get("label_count") or 0)
    if len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary label_count {expected}")
    return summary, labels


def _load_canonical_pfill(canonical_pfill_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = canonical_pfill_run / "canonical_pfill_outcome_summary.json"
    if not summary_path.exists():
        summary_path = canonical_pfill_run / "pfill_outcome_summary.json"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    labels_path = canonical_pfill_run / "pfill_order_labels.jsonl"
    by_group: dict[str, dict[str, Any]] = {}
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        canonical_group_id = str(label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError(f"{labels_path} row missing canonical_group_id")
        if canonical_group_id in by_group:
            raise ValueError(f"{labels_path} duplicate canonical_group_id={canonical_group_id}")
        by_group[canonical_group_id] = label
    expected = int(summary.get("order_label_count") or summary.get("label_count") or len(by_group))
    if len(by_group) != expected:
        raise ValueError(f"{labels_path} label count {len(by_group)} != expected {expected}")
    return summary, by_group


def _load_lifecycle_source_inputs(
    lifecycle_truth_run: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path]]:
    summary = _load_hold_summary(lifecycle_truth_run, "lifecycle_truth_audit_summary.json")
    source_inputs = summary.get("source_inputs") or []
    if not isinstance(source_inputs, list) or not source_inputs:
        raise ValueError(f"{lifecycle_truth_run} missing source_inputs")
    label_lake_by_source: dict[str, Path] = {}
    join_by_source: dict[str, Path] = {}
    for source_input in source_inputs:
        source_sha = str(source_input.get("source_telemetry_sha256") or "")
        if not source_sha:
            raise ValueError("lifecycle source input missing source_telemetry_sha256")
        label_lake_run = _resolve_path(source_input.get("label_lake_run"))
        if label_lake_run is None:
            raise ValueError(f"lifecycle source input {source_sha} missing label_lake_run")
        labels_path = label_lake_run / "labels.jsonl"
        expected_hash = str(source_input.get("label_lake_labels_sha256") or "")
        if expected_hash and expected_hash != _sha256_file(labels_path):
            raise ValueError(f"{labels_path} sha256 mismatch against lifecycle truth source input")
        label_lake_by_source[source_sha] = label_lake_run
        join_run = _resolve_path(source_input.get("join_holdout_run"))
        if join_run is not None:
            join_by_source[source_sha] = join_run
    return summary, label_lake_by_source, join_by_source


def _load_join_and_fill_indexes(
    label_lake_by_source: dict[str, Path],
    join_by_source: dict[str, Path],
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    joined_by_order: dict[tuple[str, int], list[dict[str, Any]]] = {}
    fills_by_id: dict[str, list[dict[str, Any]]] = {}
    source_summaries: list[dict[str, Any]] = []
    for source_sha, label_lake_run in sorted(label_lake_by_source.items()):
        join_run = join_by_source.get(source_sha)
        if join_run is None:
            raise ValueError(f"source {source_sha} missing join_holdout_run in lifecycle truth input")
        join_summary = _load_hold_summary(join_run, "join_holdout_summary.json")
        if str(join_summary.get("source_telemetry_sha256") or "") != source_sha:
            raise ValueError(f"{join_run} source_telemetry_sha256 mismatch")
        joined_labels_path = join_run / "joined_labels.jsonl"
        for _, row in _iter_jsonl(joined_labels_path):
            if row.get("label_type") != "DETERMINISTIC_JOIN_LABEL":
                continue
            _check_unsafe(row, joined_labels_path, label="joined label")
            order_label_seq = _safe_int(row.get("order_label_seq"))
            if order_label_seq is not None:
                joined_by_order.setdefault((source_sha, order_label_seq), []).append(row)

        observed_run = _resolve_path(join_summary.get("observed_run"))
        if observed_run is None:
            raise ValueError(f"{join_run} summary missing observed_run")
        observed_summary = _load_hold_summary(observed_run, "observed_label_summary.json")
        observed_labels_path = observed_run / "labels.jsonl"
        for _, row in _iter_jsonl(observed_labels_path):
            if row.get("label_type") != "OBSERVED_FILL_LABEL":
                continue
            _check_unsafe(row, observed_labels_path, label="observed fill label")
            fill_id = str(row.get("fill_id") or "")
            if fill_id:
                fills_by_id.setdefault(fill_id, []).append(row)
        source_summaries.append({
            "source_telemetry_sha256": source_sha,
            "label_lake_run": str(label_lake_run),
            "label_lake_labels_sha256": _sha256_file(label_lake_run / "labels.jsonl"),
            "join_holdout_run": str(join_run),
            "join_holdout_summary_sha256": _sha256_file(join_run / "join_holdout_summary.json"),
            "joined_labels_sha256": _sha256_file(joined_labels_path),
            "observed_run": str(observed_run),
            "observed_label_summary_sha256": _sha256_file(observed_run / "observed_label_summary.json"),
            "observed_labels_sha256": _sha256_file(observed_labels_path),
            "observed_gate_reason": observed_summary.get("gate_reason"),
        })
    return joined_by_order, fills_by_id, source_summaries


def _recover_source_ticks(
    *,
    order_source_t: int | None,
    joined_rows: list[dict[str, Any]],
    fills_by_id: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, int | None, dict[str, Any]]:
    if order_source_t is None:
        return "MISSING_ORDER_SOURCE_TICK", None, {"matched_fill_count": 0}
    fill_source_values: list[int] = []
    matched_fill_count = 0
    ambiguous_fill_ids = 0
    missing_fill_source_tick = 0
    for joined in joined_rows:
        fill_id = str(joined.get("fill_id") or "")
        if not fill_id:
            continue
        observed = fills_by_id.get(fill_id, [])
        if not observed:
            continue
        matched_fill_count += 1
        source_ticks = sorted({tick for row in observed if (tick := _safe_int(row.get("source_t"))) is not None})
        if len(source_ticks) > 1:
            ambiguous_fill_ids += 1
        if not source_ticks:
            missing_fill_source_tick += 1
            continue
        fill_source_values.extend(source_ticks)
    if ambiguous_fill_ids > 0:
        return "AMBIGUOUS_FILL_MATCH", None, {
            "matched_fill_count": matched_fill_count,
            "ambiguous_fill_id_count": ambiguous_fill_ids,
        }
    if not fill_source_values:
        return "MISSING_FILL_SOURCE_TICK", None, {
            "matched_fill_count": matched_fill_count,
            "missing_fill_source_tick_count": missing_fill_source_tick,
        }
    nonnegative = [value - order_source_t for value in fill_source_values if value >= order_source_t]
    if not nonnegative:
        return "NEGATIVE_HORIZON", None, {
            "matched_fill_count": matched_fill_count,
            "earliest_fill_source_t": min(fill_source_values),
            "order_source_t": order_source_t,
        }
    return "RECOVERED_SOURCE_TICKS", min(nonnegative), {
        "matched_fill_count": matched_fill_count,
        "candidate_fill_source_tick_count": len(fill_source_values),
        "multiple_fill_candidate_count": max(0, len(set(fill_source_values)) - 1),
        "earliest_fill_source_t": min(fill_source_values),
    }


def _recover_exchange_ms(
    *,
    canonical_label: dict[str, Any] | None,
    joined_rows: list[dict[str, Any]],
) -> tuple[str | None, int | None, dict[str, Any]]:
    order_ms = _first_int(
        (canonical_label or {}).get("order_time_ms"),
        (canonical_label or {}).get("order_timestamp_ms"),
        (canonical_label or {}).get("timestamp_exchange_ms"),
        (canonical_label or {}).get("created_at_ms"),
        *[row.get("order_time_ms") for row in joined_rows],
        *[row.get("order_timestamp_ms") for row in joined_rows],
    )
    fill_ms_values = [
        value
        for value in [_safe_int(row.get("fill_time_ms")) for row in joined_rows]
        if value is not None
    ]
    if order_ms is None or not fill_ms_values:
        return "INCOMPATIBLE_TIMEBASE", None, {
            "order_exchange_ms_present": order_ms is not None,
            "fill_exchange_ms_count": len(fill_ms_values),
        }
    nonnegative = [value - order_ms for value in fill_ms_values if value >= order_ms]
    if not nonnegative:
        return "NEGATIVE_HORIZON", None, {
            "order_exchange_ms": order_ms,
            "earliest_fill_exchange_ms": min(fill_ms_values),
        }
    return "RECOVERED_EXCHANGE_MS", min(nonnegative), {
        "order_exchange_ms_present": True,
        "fill_exchange_ms_count": len(fill_ms_values),
        "earliest_fill_exchange_ms": min(fill_ms_values),
    }


def _build_recovery_label(
    *,
    seq: int,
    run_id: str,
    timestamp_ns: int,
    feature_label: dict[str, Any],
    canonical_label: dict[str, Any] | None,
    joined_rows: list[dict[str, Any]],
    fills_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    canonical_group_id = str(feature_label.get("canonical_group_id") or "")
    outcome_status = str(feature_label.get("outcome_status") or "UNKNOWN")
    input_horizon = _safe_int(feature_label.get("observed_horizon_source_ticks"))
    order_source_t = _first_int(
        (canonical_label or {}).get("order_source_t"),
        *[row.get("order_source_t") for row in joined_rows],
    )
    recovered_source_ticks: int | None = None
    recovered_exchange_ms: int | None = None
    recovery_timebase = "NONE"
    source_tick_detail: dict[str, Any] = {}
    exchange_ms_detail: dict[str, Any] = {}

    if input_horizon is not None:
        recovery_status = "PRESERVED_EXISTING_SOURCE_TICKS"
        recovered_source_ticks = input_horizon
        recovery_timebase = "SOURCE_TICKS"
    elif outcome_status != "OBSERVED_FILLED":
        recovery_status = "NOT_FILLED_NOT_APPLICABLE"
    elif canonical_label is None or not joined_rows:
        recovery_status = "MISSING_JOIN"
    else:
        source_status, source_value, source_tick_detail = _recover_source_ticks(
            order_source_t=order_source_t,
            joined_rows=joined_rows,
            fills_by_id=fills_by_id,
        )
        if source_status == "RECOVERED_SOURCE_TICKS":
            recovery_status = source_status
            recovered_source_ticks = source_value
            recovery_timebase = "SOURCE_TICKS"
        else:
            exchange_status, exchange_value, exchange_ms_detail = _recover_exchange_ms(
                canonical_label=canonical_label,
                joined_rows=joined_rows,
            )
            if exchange_status == "RECOVERED_EXCHANGE_MS":
                recovery_status = exchange_status
                recovered_exchange_ms = exchange_value
                recovery_timebase = "EXCHANGE_MS"
            else:
                recovery_status = source_status or exchange_status or "UNRECOVERED"
                if recovery_status == "MISSING_FILL_SOURCE_TICK" and exchange_status == "INCOMPATIBLE_TIMEBASE":
                    recovery_status = "INCOMPATIBLE_TIMEBASE"

    effective_source_ticks = input_horizon if input_horizon is not None else recovered_source_ticks
    return {
        "schema_version": 1,
        "label_type": "PHASE51K_FILLED_HORIZON_TIMEBASE_RECOVERY_LABEL",
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
        "raw_identifier_redaction_status": "PASS",
        "canonical_group_id": canonical_group_id,
        "canonical_order_key": feature_label.get("canonical_order_key"),
        "source_telemetry_sha256": feature_label.get("source_telemetry_sha256"),
        "venue_id": feature_label.get("venue_id") or "UNKNOWN",
        "side": _canonical_side(feature_label.get("side")),
        "order_holdout_split": str(feature_label.get("order_holdout_split") or "UNKNOWN").upper(),
        "outcome_status": outcome_status,
        "p_fill_outcome": feature_label.get("p_fill_outcome"),
        "fill_count": _safe_int((canonical_label or {}).get("fill_count")) or _safe_int(feature_label.get("fill_count")) or 0,
        "input_observed_horizon_source_ticks": input_horizon,
        "recovery_status": recovery_status,
        "recovery_timebase": recovery_timebase,
        "recovered_observed_horizon_source_ticks": (
            recovered_source_ticks if input_horizon is None else None
        ),
        "recovered_observed_horizon_exchange_ms": recovered_exchange_ms,
        "effective_observed_horizon_source_ticks": effective_source_ticks,
        "effective_observed_horizon_source_ticks_available": effective_source_ticks is not None,
        "exchange_ms_horizon_available": recovered_exchange_ms is not None,
        "order_source_t": order_source_t,
        "joined_fill_label_count": len(joined_rows),
        "matched_fill_count": int(source_tick_detail.get("matched_fill_count") or 0),
        "candidate_fill_source_tick_count": int(source_tick_detail.get("candidate_fill_source_tick_count") or 0),
        "multiple_fill_candidate_count": int(source_tick_detail.get("multiple_fill_candidate_count") or 0),
        "ambiguous_fill_id_count": int(source_tick_detail.get("ambiguous_fill_id_count") or 0),
        "missing_fill_source_tick_count": int(source_tick_detail.get("missing_fill_source_tick_count") or 0),
        "order_exchange_ms_present": bool(exchange_ms_detail.get("order_exchange_ms_present", False)),
        "fill_exchange_ms_count": int(exchange_ms_detail.get("fill_exchange_ms_count") or 0),
        "timebase_isolated": True,
        "exchange_ms_not_written_to_source_ticks": recovered_exchange_ms is not None and effective_source_ticks is None,
    }


def _empty_bucket_counts() -> dict[str, int]:
    return {
        "label_count": 0,
        "filled_label_count": 0,
        "input_filled_missing_horizon_count": 0,
        "preserved_existing_source_tick_count": 0,
        "recovered_source_tick_count": 0,
        "recovered_exchange_ms_count": 0,
        "exchange_ms_only_count": 0,
        "still_missing_filled_horizon_count": 0,
        "missing_join_count": 0,
        "missing_fill_source_tick_count": 0,
        "incompatible_timebase_count": 0,
        "negative_horizon_count": 0,
        "ambiguous_fill_match_count": 0,
        "not_filled_not_applicable_count": 0,
    }


def _add_to_bucket(counts: dict[str, int], label: dict[str, Any]) -> None:
    counts["label_count"] += 1
    status = str(label.get("recovery_status") or "UNKNOWN")
    is_filled = label.get("outcome_status") == "OBSERVED_FILLED"
    if is_filled:
        counts["filled_label_count"] += 1
        if label.get("input_observed_horizon_source_ticks") is None:
            counts["input_filled_missing_horizon_count"] += 1
    if status == "PRESERVED_EXISTING_SOURCE_TICKS":
        counts["preserved_existing_source_tick_count"] += 1
    elif status == "RECOVERED_SOURCE_TICKS":
        counts["recovered_source_tick_count"] += 1
    elif status == "RECOVERED_EXCHANGE_MS":
        counts["recovered_exchange_ms_count"] += 1
        counts["exchange_ms_only_count"] += 1
    elif status == "MISSING_JOIN":
        counts["missing_join_count"] += 1
    elif status == "MISSING_FILL_SOURCE_TICK":
        counts["missing_fill_source_tick_count"] += 1
    elif status == "INCOMPATIBLE_TIMEBASE":
        counts["incompatible_timebase_count"] += 1
    elif status == "NEGATIVE_HORIZON":
        counts["negative_horizon_count"] += 1
    elif status == "AMBIGUOUS_FILL_MATCH":
        counts["ambiguous_fill_match_count"] += 1
    elif status == "NOT_FILLED_NOT_APPLICABLE":
        counts["not_filled_not_applicable_count"] += 1
    if is_filled and label.get("effective_observed_horizon_source_ticks") is None:
        counts["still_missing_filled_horizon_count"] += 1


def _bucket_entries(label: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    venue = str(label.get("venue_id") or "UNKNOWN")
    side = str(label.get("side") or "UNKNOWN")
    return [
        ("GLOBAL", {"scope": "GLOBAL"}),
        (f"VENUE:{venue}", {"venue_id": venue}),
        (f"SIDE:{side}", {"side": side}),
        (f"VENUE_SIDE:{venue}:{side}", {"venue_id": venue, "side": side}),
    ]


def _build_bucket_records(labels: list[dict[str, Any]], *, run_id: str, timestamp_ns: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for label in labels:
        for bucket_id, dimensions in _bucket_entries(label):
            bucket = buckets.setdefault(bucket_id, {"dimensions": dimensions, "counts": _empty_bucket_counts()})
            _add_to_bucket(bucket["counts"], label)
    records: list[dict[str, Any]] = []
    for seq, (bucket_id, bucket) in enumerate(sorted(buckets.items()), start=1):
        counts = bucket["counts"]
        reasons = ["requires_phase51h_phase51i_replay_before_model_training"]
        if counts["still_missing_filled_horizon_count"] > 0:
            reasons.insert(0, "filled_horizon_source_tick_missing")
        if counts["exchange_ms_only_count"] > 0:
            reasons.insert(0, "filled_horizon_exchange_ms_only")
        records.append({
            "schema_version": 1,
            "label_type": "PHASE51K_FILLED_HORIZON_TIMEBASE_BUCKET",
            "bucket_seq": seq,
            "timestamp_local_ns": timestamp_ns + seq,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "bucket_id": bucket_id,
            "bucket_dimensions": bucket["dimensions"],
            "gate_status": "HOLD",
            "gate_reasons": sorted(set(reasons)),
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
            **counts,
        })
    return records


def _summary_gate_reason(global_counts: dict[str, int]) -> str:
    if global_counts["still_missing_filled_horizon_count"] > 0:
        return "phase51k_filled_horizon_timebase_partial"
    if global_counts["exchange_ms_only_count"] > 0:
        return "phase51k_filled_horizon_exchange_ms_only_requires_board_review"
    return "phase51k_filled_horizon_source_tick_complete_nonlive_hold"


def build_filled_horizon_timebase_recovery(
    *,
    feature_audit_run: Path,
    canonical_pfill_run: Path,
    lifecycle_truth_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51K-FILLED-HORIZON-TIMEBASE-RECOVERY-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    feature_audit_run = _resolve_path(feature_audit_run)
    canonical_pfill_run = _resolve_path(canonical_pfill_run)
    lifecycle_truth_run = _resolve_path(lifecycle_truth_run)
    assert feature_audit_run is not None and canonical_pfill_run is not None and lifecycle_truth_run is not None

    feature_summary, feature_labels = _load_feature_audit(feature_audit_run)
    canonical_summary, canonical_by_group = _load_canonical_pfill(canonical_pfill_run)
    lifecycle_summary, label_lake_by_source, join_by_source = _load_lifecycle_source_inputs(lifecycle_truth_run)
    joined_by_order, fills_by_id, source_summaries = _load_join_and_fill_indexes(label_lake_by_source, join_by_source)

    recovery_labels: list[dict[str, Any]] = []
    for seq, feature_label in enumerate(feature_labels, start=1):
        canonical_group_id = str(feature_label.get("canonical_group_id") or "")
        canonical_label = canonical_by_group.get(canonical_group_id)
        source_sha = str(feature_label.get("source_telemetry_sha256") or "")
        order_label_seq = _safe_int((canonical_label or {}).get("order_label_seq"))
        joined_rows = joined_by_order.get((source_sha, order_label_seq), []) if order_label_seq is not None else []
        label = _build_recovery_label(
            seq=seq,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            feature_label=feature_label,
            canonical_label=canonical_label,
            joined_rows=joined_rows,
            fills_by_id=fills_by_id,
        )
        _check_no_raw_identifiers(label, out_dir / "filled_horizon_timebase_recovery_labels.jsonl", label="recovery label")
        recovery_labels.append(label)

    bucket_records = _build_bucket_records(recovery_labels, run_id=run_id, timestamp_ns=timestamp_ns)
    global_bucket = next(record for record in bucket_records if record["bucket_id"] == "GLOBAL")
    global_counts = {key: int(global_bucket.get(key) or 0) for key in _empty_bucket_counts()}
    status_counts: dict[str, int] = {}
    for label in recovery_labels:
        status = str(label.get("recovery_status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    labels_path = out_dir / "filled_horizon_timebase_recovery_labels.jsonl"
    buckets_path = out_dir / "filled_horizon_timebase_recovery_buckets.jsonl"
    summary_path = out_dir / "filled_horizon_timebase_recovery_summary.json"
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": _summary_gate_reason(global_counts),
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
        "raw_identifier_redaction_status": "PASS",
        "feature_audit_run": str(feature_audit_run),
        "canonical_pfill_run": str(canonical_pfill_run),
        "lifecycle_truth_run": str(lifecycle_truth_run),
        "feature_audit_summary_sha256": _sha256_file(feature_audit_run / "pfill_feature_audit_summary.json"),
        "feature_coverage_labels_sha256": _sha256_file(feature_audit_run / "pfill_feature_coverage_labels.jsonl"),
        "canonical_pfill_summary_hash": _stable_hash(canonical_summary),
        "lifecycle_truth_summary_hash": _stable_hash(lifecycle_summary),
        "source_inputs": source_summaries,
        "source_telemetry_sha256_list": sorted(label_lake_by_source),
        "label_count": len(recovery_labels),
        "bucket_count": len(bucket_records),
        "recovery_status_counts": dict(sorted(status_counts.items())),
        **global_counts,
    }
    _write_jsonl(labels_path, recovery_labels)
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
    parser.add_argument("--feature-audit-run", type=Path, required=True)
    parser.add_argument("--canonical-pfill-run", type=Path, required=True)
    parser.add_argument("--lifecycle-truth-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_filled_horizon_timebase_recovery(
            feature_audit_run=args.feature_audit_run,
            canonical_pfill_run=args.canonical_pfill_run,
            lifecycle_truth_run=args.lifecycle_truth_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51k_filled_horizon_timebase_recovery: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51k_filled_horizon_timebase_recovery: wrote {out_dir}")
    print("phase51k_filled_horizon_timebase_recovery: status HOLD (filled horizon evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
