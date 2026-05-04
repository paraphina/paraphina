#!/usr/bin/env python3
"""Build Phase 5.1n event-time Lighter native-limit alignment labels.

This is a read-only, non-live evidence gate. It maps Lighter P_fill labels to
historical source telemetry event time and nearest historical Lighter account
snapshot records. It does not submit orders, query live exchange state, infer
sendTx/REST headroom from documentation, approve model training, or approve
live/canary/capital/risk changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51n_lighter_native_limit_time_alignment"
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
SNAPSHOT_RE = re.compile(
    r"Lighter account snapshot\s+seq=(?P<seq>\d+)\s+ts=(?P<ts>\d+).*?"
    r"total_order_count=(?P<total>\d+)\s+pending_order_count=(?P<pending>\d+)"
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
    return path if path.is_absolute() else ROOT / path


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


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, record


def _iter_json_stream(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            source = line.strip()
            if not source:
                continue
            pos = 0
            while pos < len(source):
                record, end = decoder.raw_decode(source, pos)
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_no}")
                yield line_no, record
                pos = end
                while pos < len(source) and source[pos].isspace():
                    pos += 1


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


def _event_time_ms(record: dict[str, Any]) -> int | None:
    for key in ("timestamp_local_ms", "timestamp_ms", "exchange_timestamp_ms", "kf_last_update_ms"):
        value = _safe_int(record.get(key))
        if value is not None:
            return value
    treasury = record.get("treasury_guidance")
    if isinstance(treasury, dict):
        value = _safe_int(treasury.get("as_of_ms"))
        if value is not None:
            return value
    return None


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _load_phase51b_context(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {
            "run_dir": None,
            "run_id": None,
            "acceptance_sha256": None,
            "telemetry_sha256": None,
            "limitations": ["phase51b_native_limits_run_not_supplied"],
            "active_order_cap_account": None,
            "active_order_cap_market": None,
            "sendtx_per_minute_limit": None,
            "sendtx_per_minute_remaining": None,
            "rest_requests_per_minute_limit": None,
            "rest_requests_per_minute_remaining": None,
            "weighted_requests_per_minute_limit": None,
            "weighted_requests_per_minute_remaining": None,
        }
    run_dir = _resolve_path(run_dir)
    acceptance_path = run_dir / "phase51b_acceptance.json"
    telemetry_path = run_dir / "telemetry.jsonl"
    acceptance = _load_json(acceptance_path)
    _check_unsafe(acceptance, acceptance_path, label="acceptance")
    active_order_cap_account = None
    active_order_cap_market = None
    sendtx_per_minute_limit = None
    sendtx_per_minute_remaining = None
    rest_requests_per_minute_limit = None
    rest_requests_per_minute_remaining = None
    weighted_requests_per_minute_limit = None
    weighted_requests_per_minute_remaining = None
    for _, event in _iter_jsonl(telemetry_path):
        _check_unsafe(event, telemetry_path, label="telemetry")
        if event.get("event_type") == "V2_LIGHTER_ACTIVE_ORDERS":
            total = _safe_int(event.get("active_orders_count_total"))
            account_headroom = _safe_int(event.get("active_order_headroom_account"))
            if total is not None and account_headroom is not None:
                active_order_cap_account = total + account_headroom
            market_count = _safe_int(event.get("active_orders_count_market"))
            market_headroom = _safe_int(event.get("active_order_headroom_market"))
            if market_count is not None and market_headroom is not None:
                active_order_cap_market = market_count + market_headroom
        elif event.get("event_type") == "V2_LIGHTER_ACCOUNT_LIMITS":
            sendtx_per_minute_limit = _safe_int(event.get("sendtx_per_minute_limit"))
            sendtx_per_minute_remaining = _safe_int(event.get("sendtx_per_minute_remaining"))
            rest_requests_per_minute_limit = _safe_int(event.get("rest_requests_per_minute_limit"))
            rest_requests_per_minute_remaining = _safe_int(event.get("rest_requests_per_minute_remaining"))
            weighted_requests_per_minute_limit = _safe_int(event.get("weighted_requests_per_minute_limit"))
            weighted_requests_per_minute_remaining = _safe_int(event.get("weighted_requests_per_minute_remaining"))
    limitations = list(acceptance.get("limitations") or [])
    if sendtx_per_minute_limit is None and "lighter_sendtx_limit_not_observed" not in limitations:
        limitations.append("lighter_sendtx_limit_not_observed")
    if sendtx_per_minute_remaining is None and "lighter_sendtx_remaining_not_observed" not in limitations:
        limitations.append("lighter_sendtx_remaining_not_observed")
    if (
        rest_requests_per_minute_limit is None
        and weighted_requests_per_minute_limit is None
        and "lighter_rest_or_weighted_limit_not_observed" not in limitations
    ):
        limitations.append("lighter_rest_or_weighted_limit_not_observed")
    if (
        rest_requests_per_minute_remaining is None
        and weighted_requests_per_minute_remaining is None
        and "lighter_rest_or_weighted_remaining_not_observed" not in limitations
    ):
        limitations.append("lighter_rest_or_weighted_remaining_not_observed")
    return {
        "run_dir": str(run_dir),
        "run_id": acceptance.get("run_id"),
        "acceptance_sha256": _sha256_file(acceptance_path),
        "telemetry_sha256": _sha256_file(telemetry_path),
        "limitations": limitations,
        "active_order_cap_account": active_order_cap_account,
        "active_order_cap_market": active_order_cap_market,
        "sendtx_per_minute_limit": sendtx_per_minute_limit,
        "sendtx_per_minute_remaining": sendtx_per_minute_remaining,
        "rest_requests_per_minute_limit": rest_requests_per_minute_limit,
        "rest_requests_per_minute_remaining": rest_requests_per_minute_remaining,
        "weighted_requests_per_minute_limit": weighted_requests_per_minute_limit,
        "weighted_requests_per_minute_remaining": weighted_requests_per_minute_remaining,
    }


def _load_pfill_labels(pfill_outcome_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = pfill_outcome_run / "pfill_outcome_summary.json"
    labels_path = pfill_outcome_run / "pfill_order_labels.jsonl"
    summary = _load_json(summary_path)
    _check_unsafe(summary, summary_path, label="summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        labels.append(label)
    return summary, labels


def _load_source_times(source_paths: list[Path]) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    by_source: dict[str, dict[int, dict[str, Any]]] = {}
    infos: list[dict[str, Any]] = []
    for path in source_paths:
        path = _resolve_path(path)
        source_sha = _sha256_file(path)
        index: dict[int, dict[str, Any]] = {}
        for line_no, record in _iter_json_stream(path):
            tick = _safe_int(record.get("t"))
            if tick is None:
                continue
            event_time = _event_time_ms(record)
            if event_time is None:
                continue
            index[tick] = {
                "source_line": line_no,
                "source_t": tick,
                "event_time_ms": event_time,
                "source_time_field": "event_time_ms_from_telemetry",
            }
        by_source[source_sha] = index
        infos.append({
            "path": str(path),
            "sha256": source_sha,
            "event_time_count": len(index),
        })
    return by_source, infos


def _load_snapshots(snapshot_paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshots: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for path in snapshot_paths:
        path = _resolve_path(path)
        path_sha = _sha256_file(path)
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                match = SNAPSHOT_RE.search(line)
                if not match:
                    continue
                count += 1
                snapshots.append({
                    "snapshot_ts_ms": int(match.group("ts")),
                    "snapshot_seq_hash": _stable_hash(match.group("seq")),
                    "snapshot_source_path": str(path),
                    "snapshot_source_sha256": path_sha,
                    "snapshot_source_line": line_no,
                    "native_active_orders_count_total": int(match.group("total")),
                    "native_pending_orders_count_total": int(match.group("pending")),
                })
        infos.append({"path": str(path), "sha256": path_sha, "snapshot_count": count})
    snapshots.sort(key=lambda row: int(row["snapshot_ts_ms"]))
    return snapshots, infos


def _nearest_snapshot(
    snapshots: list[dict[str, Any]],
    snapshot_times: list[int],
    event_time_ms: int,
) -> dict[str, Any] | None:
    if not snapshots:
        return None
    idx = bisect_left(snapshot_times, event_time_ms)
    candidates: list[dict[str, Any]] = []
    if idx < len(snapshots):
        candidates.append(snapshots[idx])
    if idx > 0:
        candidates.append(snapshots[idx - 1])
    return min(candidates, key=lambda row: abs(int(row["snapshot_ts_ms"]) - event_time_ms))


def _base_label(run_id: str, seq: int, timestamp_ns: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": "PHASE51N_LIGHTER_NATIVE_LIMIT_TIME_ALIGNMENT_LABEL",
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _all_native_limit_pressure_dimensions_observed(record: dict[str, Any], context: dict[str, Any]) -> bool:
    has_active = (
        record.get("native_active_order_headroom_account") is not None
        and record.get("native_active_order_headroom_market") is not None
    )
    has_sendtx = (
        context.get("sendtx_per_minute_limit") is not None
        and context.get("sendtx_per_minute_remaining") is not None
    )
    has_rest_or_weighted = (
        context.get("rest_requests_per_minute_limit") is not None
        and context.get("rest_requests_per_minute_remaining") is not None
    ) or (
        context.get("weighted_requests_per_minute_limit") is not None
        and context.get("weighted_requests_per_minute_remaining") is not None
    )
    return (
        record.get("native_limit_time_alignment_status") == "EVENT_TIME_ALIGNED"
        and has_active
        and has_sendtx
        and has_rest_or_weighted
    )


def _forward_native_limit_pressure_source_row(
    record: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    source_row = {
        "schema_version": 1,
        "source": "phase51n_lighter_native_limit_time_alignment",
        "source_type": "LIGHTER_LIMITS_AT_DECISION_TIME",
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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
        "venue_id": "lighter",
        "canonical_group_id": record.get("canonical_group_id"),
        "order_key": record.get("order_key"),
        "active_order_headroom_account": record.get("native_active_order_headroom_account"),
        "active_order_headroom_market": record.get("native_active_order_headroom_market"),
        "sendtx_per_minute_limit": context.get("sendtx_per_minute_limit"),
        "sendtx_per_minute_remaining": context.get("sendtx_per_minute_remaining"),
        "rest_requests_per_minute_limit": context.get("rest_requests_per_minute_limit"),
        "rest_requests_per_minute_remaining": context.get("rest_requests_per_minute_remaining"),
        "weighted_requests_per_minute_limit": context.get("weighted_requests_per_minute_limit"),
        "weighted_requests_per_minute_remaining": context.get("weighted_requests_per_minute_remaining"),
        "native_limit_event_time_status": record.get("native_limit_time_alignment_status"),
        "native_limit_staleness_ms": record.get("snapshot_age_ms_abs"),
        "source_event_time_ms": record.get("source_event_time_ms"),
        "snapshot_ts_ms": record.get("snapshot_ts_ms"),
        "phase51b_native_run_id": record.get("phase51b_native_run_id"),
        "source_alignment_label_sha256": _stable_hash(record),
    }
    return {key: value for key, value in source_row.items() if value is not None}


def _phase51v_manifest(forward_source_path: Path) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_model_training": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "sources": [
            {
                "source_id": "phase51n_lighter_forward_native_limit_pressure",
                "venue_id": "lighter",
                "path": str(forward_source_path),
            }
        ],
        "source_links": [],
    }


def build_lighter_native_limit_time_alignment(
    *,
    pfill_outcome_run: Path,
    source_telemetry: list[Path],
    lighter_snapshot_log: list[Path],
    phase51b_native_run: Path | None,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    max_snapshot_age_ms: int,
) -> Path:
    run_id = run_id or f"PHASE51N-LIGHTER-NATIVE-LIMIT-TIME-ALIGNMENT-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    pfill_outcome_run = _resolve_path(pfill_outcome_run)
    phase51b_context = _load_phase51b_context(phase51b_native_run)
    pfill_summary, pfill_labels = _load_pfill_labels(pfill_outcome_run)
    source_times, source_infos = _load_source_times(source_telemetry)
    snapshots, snapshot_infos = _load_snapshots(lighter_snapshot_log)
    snapshot_times = [int(row["snapshot_ts_ms"]) for row in snapshots]

    account_cap = phase51b_context.get("active_order_cap_account")
    market_cap = phase51b_context.get("active_order_cap_market")
    sendtx_remaining = phase51b_context.get("sendtx_per_minute_remaining")
    rest_remaining = phase51b_context.get("rest_requests_per_minute_remaining")
    weighted_remaining = phase51b_context.get("weighted_requests_per_minute_remaining")

    records: list[dict[str, Any]] = []
    for seq, label in enumerate(pfill_labels, start=1):
        record = _base_label(run_id, seq, timestamp_ns)
        source_sha = str(label.get("source_telemetry_sha256") or pfill_summary.get("source_telemetry_sha256") or "")
        venue = str(label.get("venue_id") or "UNKNOWN").lower()
        order_source_t = _safe_int(label.get("order_source_t"))
        record.update({
            "source": "phase51n_lighter_native_limit_time_alignment",
            "pfill_outcome_run": str(pfill_outcome_run),
            "phase51b_native_run": phase51b_context.get("run_dir"),
            "phase51b_native_run_id": phase51b_context.get("run_id"),
            "source_telemetry_sha256": source_sha,
            "order_key": label.get("order_key"),
            "canonical_group_id": label.get("canonical_group_id"),
            "order_label_seq": label.get("order_label_seq"),
            "order_source_line": label.get("order_source_line"),
            "order_source_t": order_source_t,
            "venue_id": label.get("venue_id"),
            "native_limit_context_source": "LIGHTER_ACCOUNT_SNAPSHOT_LOG",
            "native_limit_time_alignment_status": "NOT_APPLICABLE_NON_LIGHTER",
            "native_limit_alignment_hold_reason": "not_lighter_venue",
            "native_limit_all_pressure_dimensions_observed": False,
            "source_event_time_ms": None,
            "source_time_field": None,
            "snapshot_ts_ms": None,
            "snapshot_age_ms_abs": None,
            "snapshot_source_sha256": None,
            "snapshot_source_line": None,
            "snapshot_seq_hash": None,
            "native_active_orders_count_total": None,
            "native_pending_orders_count_total": None,
            "native_active_order_headroom_account": None,
            "native_active_order_headroom_market": None,
            "native_sendtx_per_minute_remaining": sendtx_remaining,
            "native_rest_requests_per_minute_remaining": rest_remaining,
            "native_weighted_requests_per_minute_remaining": weighted_remaining,
            "native_active_order_limit_source": "PHASE51B_OFFICIAL_CAP_CONTEXT" if account_cap is not None else None,
            "native_active_order_limit_conflicts": [],
            "native_limit_limitations": phase51b_context.get("limitations") or [],
        })
        if venue != "lighter":
            records.append(record)
            continue
        source_index = source_times.get(source_sha)
        if source_index is None:
            record["native_limit_time_alignment_status"] = "MISSING_SOURCE_TELEMETRY"
            record["native_limit_alignment_hold_reason"] = "source_telemetry_sha_not_supplied"
            records.append(record)
            continue
        if order_source_t is None or order_source_t not in source_index:
            record["native_limit_time_alignment_status"] = "MISSING_SOURCE_EVENT_TIME"
            record["native_limit_alignment_hold_reason"] = "order_source_t_not_found_in_source_telemetry"
            records.append(record)
            continue
        source_time = source_index[order_source_t]
        event_time_ms = int(source_time["event_time_ms"])
        record["source_event_time_ms"] = event_time_ms
        record["source_time_field"] = source_time["source_time_field"]
        nearest = _nearest_snapshot(snapshots, snapshot_times, event_time_ms)
        if nearest is None:
            record["native_limit_time_alignment_status"] = "MISSING_LIGHTER_ACCOUNT_SNAPSHOT"
            record["native_limit_alignment_hold_reason"] = "no_lighter_snapshot_log_records"
            records.append(record)
            continue
        snapshot_age = abs(int(nearest["snapshot_ts_ms"]) - event_time_ms)
        record.update({
            "snapshot_ts_ms": nearest["snapshot_ts_ms"],
            "snapshot_age_ms_abs": snapshot_age,
            "snapshot_source_sha256": nearest["snapshot_source_sha256"],
            "snapshot_source_line": nearest["snapshot_source_line"],
            "snapshot_seq_hash": nearest["snapshot_seq_hash"],
            "native_active_orders_count_total": nearest["native_active_orders_count_total"],
            "native_pending_orders_count_total": nearest["native_pending_orders_count_total"],
        })
        if account_cap is not None:
            record["native_active_order_headroom_account"] = (
                int(account_cap) - int(nearest["native_active_orders_count_total"])
            )
        if market_cap is not None:
            record["native_active_order_headroom_market"] = (
                int(market_cap) - int(nearest["native_active_orders_count_total"])
            )
        if snapshot_age > max_snapshot_age_ms:
            record["native_limit_time_alignment_status"] = "STALE_LIGHTER_ACCOUNT_SNAPSHOT"
            record["native_limit_alignment_hold_reason"] = "nearest_snapshot_exceeds_max_age_ms"
        elif account_cap is None:
            record["native_limit_time_alignment_status"] = "EVENT_TIME_ALIGNED_CAP_UNKNOWN"
            record["native_limit_alignment_hold_reason"] = "event_time_snapshot_present_but_account_cap_unknown"
        else:
            record["native_limit_time_alignment_status"] = "EVENT_TIME_ALIGNED"
            if _all_native_limit_pressure_dimensions_observed(record, phase51b_context):
                record["native_limit_alignment_hold_reason"] = "requires_queue_reset_calibration_and_board_review"
                record["native_limit_all_pressure_dimensions_observed"] = True
            else:
                record["native_limit_alignment_hold_reason"] = (
                    "event_time_active_order_headroom_observed_but_sendtx_or_rest_remaining_unobserved"
                )
        records.append(record)

    labels_path = out_dir / "lighter_native_limit_time_alignment_labels.jsonl"
    forward_source_path = out_dir / "lighter_forward_native_limit_pressure_snapshot.jsonl"
    phase51v_manifest_path = out_dir / "phase51v_lighter_native_limit_manifest.generated.json"
    summary_path = out_dir / "lighter_native_limit_time_alignment_summary.json"
    forward_source_records = [
        _forward_native_limit_pressure_source_row(record, phase51b_context)
        for record in records
        if _all_native_limit_pressure_dimensions_observed(record, phase51b_context)
    ]
    _write_jsonl(labels_path, records)
    _write_jsonl(forward_source_path, forward_source_records)
    _write_json(phase51v_manifest_path, _phase51v_manifest(forward_source_path))
    status_counts = _status_counts(records, "native_limit_time_alignment_status")
    aligned = status_counts.get("EVENT_TIME_ALIGNED", 0)
    fully_observed = sum(1 for record in records if record.get("native_limit_all_pressure_dimensions_observed") is True)
    lighter_count = sum(1 for record in records if str(record.get("venue_id") or "").lower() == "lighter")
    partial_count = lighter_count - fully_observed
    gate_reason = (
        "phase51n_lighter_native_limit_all_pressure_dimensions_observed"
        if lighter_count > 0 and partial_count == 0
        else "phase51n_lighter_native_limit_event_time_alignment_partial"
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
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
        "pfill_outcome_run": str(pfill_outcome_run),
        "pfill_outcome_summary_sha256": _sha256_file(pfill_outcome_run / "pfill_outcome_summary.json"),
        "pfill_outcome_labels_sha256": _sha256_file(pfill_outcome_run / "pfill_order_labels.jsonl"),
        "phase51b_native_run": phase51b_context.get("run_dir"),
        "phase51b_native_acceptance_sha256": phase51b_context.get("acceptance_sha256"),
        "phase51b_native_telemetry_sha256": phase51b_context.get("telemetry_sha256"),
        "source_telemetry_inputs": source_infos,
        "lighter_snapshot_log_inputs": snapshot_infos,
        "max_snapshot_age_ms": max_snapshot_age_ms,
        "label_count": len(records),
        "lighter_label_count": lighter_count,
        "native_limit_event_time_aligned_count": aligned,
        "native_limit_all_pressure_dimensions_observed_count": fully_observed,
        "native_limit_partial_or_unobserved_count": partial_count,
        "native_limit_time_alignment_status_counts": status_counts,
        "forward_native_limit_pressure_source_path": str(forward_source_path),
        "forward_native_limit_pressure_source_count": len(forward_source_records),
        "phase51v_lighter_native_limit_manifest_path": str(phase51v_manifest_path),
        "phase51v_lighter_native_limit_manifest_ready": len(forward_source_records) > 0,
        "limitations": sorted(set(phase51b_context.get("limitations") or [])),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, forward_source_path, phase51v_manifest_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(
            out_dir,
            [labels_path, forward_source_path, phase51v_manifest_path, summary_path, artifact_index_path],
        ),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfill-outcome-run", type=Path, required=True)
    parser.add_argument("--source-telemetry", type=Path, action="append", required=True)
    parser.add_argument("--lighter-snapshot-log", type=Path, action="append", required=True)
    parser.add_argument("--phase51b-native-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--max-snapshot-age-ms", type=int, default=3500)
    args = parser.parse_args()
    try:
        out_dir = build_lighter_native_limit_time_alignment(
            pfill_outcome_run=args.pfill_outcome_run,
            source_telemetry=args.source_telemetry,
            lighter_snapshot_log=args.lighter_snapshot_log,
            phase51b_native_run=args.phase51b_native_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            max_snapshot_age_ms=args.max_snapshot_age_ms,
        )
    except Exception as exc:
        print(f"phase51n_lighter_native_limit_time_alignment: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51n_lighter_native_limit_time_alignment: wrote {out_dir}")
    print("phase51n_lighter_native_limit_time_alignment: status HOLD (non-live event-time evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
