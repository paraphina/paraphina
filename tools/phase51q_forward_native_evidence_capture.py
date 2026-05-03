#!/usr/bin/env python3
"""Build Phase 5.1q forward venue-native evidence artifacts.

This HOLD-only gate consumes sanitized, forward-captured native role and
native-limit source rows. It emits canonical maker/taker evidence that can feed
Phase 5.1n recovery without inferring role from post-only, purpose, fee
schedule, or strategy intent. It also emits native-limit pressure labels for
the current Lighter-specific limit-pressure blocker.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51q_forward_native_evidence"
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
    "ask_id",
    "bid_id",
    "ask_client_id",
    "bid_client_id",
    "trade_id",
    "fill_id",
}
ALLOWED_NATIVE_ROLE_SOURCES = {
    "VENUE_NATIVE_FILL_FIELD",
    "VENUE_NATIVE_TRADE_JOIN",
    "VENUE_NATIVE_FEE_ROLE",
    "LIGHTER_TRADES_JSON",
    "HYPERLIQUID_CROSSED",
    "PARADEX_LIQUIDITY",
    "ASTER_ORDER_TRADE_UPDATE_M",
    "EXTENDED_ISTAKER",
}
VENUE_SPECIFIC_ROLE_SOURCES = {
    "LIGHTER_TRADES_JSON": "lighter",
    "HYPERLIQUID_CROSSED": "hyperliquid",
    "PARADEX_LIQUIDITY": "paradex",
    "ASTER_ORDER_TRADE_UPDATE_M": "aster",
    "EXTENDED_ISTAKER": "extended",
}
ROLE_VALUES = {"MAKER", "TAKER", "UNKNOWN"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _check_safety(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")
    raw_fields = RAW_IDENTIFIER_FIELDS & set(record)
    if raw_fields:
        raise ValueError(f"{path} has raw {label} identifier fields: {sorted(raw_fields)}")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _role_counts(value: Any) -> dict[str, int]:
    counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
    if isinstance(value, dict):
        for key in counts:
            counts[key] = max(0, _safe_int(value.get(key)) or 0)
        return counts
    role = str(value or "").upper()
    if role in ROLE_VALUES:
        counts[role] = 1
    return counts


def _known_count(counts: dict[str, int]) -> int:
    return int(counts.get("MAKER") or 0) + int(counts.get("TAKER") or 0)


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _source_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _load_observed_pfill(observed_pfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = observed_pfill_run / "pfill_outcome_summary.json"
    labels_path = observed_pfill_run / "pfill_order_labels.jsonl"
    summary = _load_json(summary_path)
    for flag in UNSAFE_TRUE_FLAGS:
        if summary.get(flag) is True:
            raise ValueError(f"{summary_path} has unsafe summary flag {flag}=true")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        for flag in UNSAFE_TRUE_FLAGS:
            if label.get(flag) is True:
                raise ValueError(f"{labels_path} has unsafe label flag {flag}=true")
        labels.append(label)
    expected = _safe_int(summary.get("order_label_count"))
    if expected is not None and len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary order_label_count {expected}")
    return summary, labels


def _normalize_role_source(row: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
    source = str(row.get("maker_taker_attribution_source") or row.get("native_role_source") or "")
    if source not in ALLOWED_NATIVE_ROLE_SOURCES:
        raise ValueError(f"{path}:{line_no} unsupported native role source {source!r}")
    venue_id = str(row.get("venue_id") or "").lower()
    expected_venue = VENUE_SPECIFIC_ROLE_SOURCES.get(source)
    if expected_venue and venue_id and venue_id != expected_venue:
        raise ValueError(f"{path}:{line_no} source {source} conflicts with venue_id={venue_id!r}")
    counts = _role_counts(row.get("maker_taker_role_counts", row.get("native_role")))
    if _known_count(counts) <= 0:
        raise ValueError(f"{path}:{line_no} missing explicit MAKER/TAKER role count")
    canonical_group_id = str(row.get("canonical_group_id") or "")
    if not canonical_group_id:
        raise ValueError(f"{path}:{line_no} missing canonical_group_id")
    return {
        "canonical_group_id": canonical_group_id,
        "venue_id": venue_id or expected_venue or "unknown",
        "maker_taker_role_counts": counts,
        "maker_taker_attribution_source": source,
        "source_record_sha256": row.get("source_record_sha256"),
        "native_role_capture_status": str(row.get("native_role_capture_status") or "OBSERVED_NATIVE_ROLE"),
    }


def _load_native_roles(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    infos: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _resolve_path(raw_path)
        count = 0
        for line_no, row in _iter_jsonl(path):
            _check_safety(row, path, label="native role source")
            record = _normalize_role_source(row, path, line_no)
            group = record["canonical_group_id"]
            if group in by_group:
                raise ValueError(f"{path}:{line_no} duplicate canonical_group_id={group}")
            by_group[group] = record
            count += 1
        infos.append({"path": str(path), "sha256": _sha256_file(path), "record_count": count})
    return by_group, infos


def _normalize_limit_source(row: dict[str, Any], path: Path, line_no: int) -> dict[str, Any]:
    canonical_group_id = str(row.get("canonical_group_id") or "")
    if not canonical_group_id:
        raise ValueError(f"{path}:{line_no} missing canonical_group_id")
    venue_id = str(row.get("venue_id") or "").lower()
    if not venue_id:
        raise ValueError(f"{path}:{line_no} missing venue_id")
    return {
        "canonical_group_id": canonical_group_id,
        "venue_id": venue_id,
        "active_order_headroom_account": _safe_int(row.get("active_order_headroom_account")),
        "active_order_headroom_market": _safe_int(row.get("active_order_headroom_market")),
        "sendtx_per_minute_limit": _safe_int(row.get("sendtx_per_minute_limit")),
        "sendtx_per_minute_remaining": _safe_int(row.get("sendtx_per_minute_remaining")),
        "rest_requests_per_minute_limit": _safe_int(row.get("rest_requests_per_minute_limit")),
        "rest_requests_per_minute_remaining": _safe_int(row.get("rest_requests_per_minute_remaining")),
        "weighted_requests_per_minute_limit": _safe_int(row.get("weighted_requests_per_minute_limit")),
        "weighted_requests_per_minute_remaining": _safe_int(row.get("weighted_requests_per_minute_remaining")),
        "native_limit_event_time_status": str(row.get("native_limit_event_time_status") or ""),
        "native_limit_staleness_ms": _safe_float(row.get("native_limit_staleness_ms")),
        "native_limit_source_sha256": row.get("native_limit_source_sha256") or row.get("source_record_sha256"),
    }


def _load_native_limits(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    infos: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _resolve_path(raw_path)
        count = 0
        for line_no, row in _iter_jsonl(path):
            _check_safety(row, path, label="native limit source")
            record = _normalize_limit_source(row, path, line_no)
            group = record["canonical_group_id"]
            if group in by_group:
                raise ValueError(f"{path}:{line_no} duplicate canonical_group_id={group}")
            by_group[group] = record
            count += 1
        infos.append({"path": str(path), "sha256": _sha256_file(path), "record_count": count})
    return by_group, infos


def _base_record(run_id: str, seq: int, timestamp_ns: int, label_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": label_type,
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


def _limit_status(venue_id: str, source: dict[str, Any] | None) -> tuple[str, str]:
    if venue_id != "lighter":
        return "NOT_APPLICABLE_NON_LIGHTER", "native_limit_pressure_gate_is_currently_lighter_specific"
    if source is None:
        return "MISSING_NATIVE_LIMIT_PRESSURE_SOURCE", "missing_event_time_native_limit_snapshot"
    has_active = (
        source.get("active_order_headroom_account") is not None
        and source.get("active_order_headroom_market") is not None
    )
    has_sendtx = (
        source.get("sendtx_per_minute_limit") is not None
        and source.get("sendtx_per_minute_remaining") is not None
    )
    has_rest_or_weighted = (
        source.get("rest_requests_per_minute_limit") is not None
        and source.get("rest_requests_per_minute_remaining") is not None
    ) or (
        source.get("weighted_requests_per_minute_limit") is not None
        and source.get("weighted_requests_per_minute_remaining") is not None
    )
    event_time_ok = source.get("native_limit_event_time_status") in {
        "EVENT_TIME_ALIGNED",
        "SNAPSHOT_AT_DECISION_TIME",
        "OBSERVED_AT_DECISION_TIME",
    }
    if has_active and has_sendtx and has_rest_or_weighted and event_time_ok:
        return "OBSERVED_NATIVE_LIMIT_PRESSURE", "event_time_native_limit_pressure_complete"
    missing = []
    if not has_active:
        missing.append("active_or_pending_order_headroom")
    if not has_sendtx:
        missing.append("sendtx_headroom")
    if not has_rest_or_weighted:
        missing.append("rest_or_weighted_request_headroom")
    if not event_time_ok:
        missing.append("event_time_alignment")
    return "PARTIAL_NATIVE_LIMIT_PRESSURE_SOURCE", ",".join(missing)


def build_forward_native_evidence(
    *,
    observed_pfill_run: Path,
    native_role_jsonl: list[Path],
    native_limit_jsonl: list[Path],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    observed_pfill_run = _resolve_path(observed_pfill_run)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_summary, labels = _load_observed_pfill(observed_pfill_run)
    role_by_group, role_source_infos = _load_native_roles(native_role_jsonl)
    limit_by_group, limit_source_infos = _load_native_limits(native_limit_jsonl)

    native_role_evidence: list[dict[str, Any]] = []
    role_labels: list[dict[str, Any]] = []
    limit_labels: list[dict[str, Any]] = []
    seq = 0
    for label in labels:
        group = str(label.get("canonical_group_id") or "")
        venue_id = str(label.get("venue_id") or "").lower()
        fill_count = _safe_int(label.get("fill_count")) or 0
        observed_counts = _role_counts(label.get("maker_taker_role_counts"))
        role_source = role_by_group.get(group)
        role_record = _base_record(
            run_id,
            seq,
            timestamp_ns,
            "PHASE51Q_FORWARD_NATIVE_ROLE_CAPTURE_LABEL",
        )
        seq += 1
        role_record.update(
            {
                "canonical_group_id": group,
                "order_key": label.get("order_key"),
                "venue_id": venue_id,
                "fill_count": fill_count,
                "input_maker_taker_role_counts": observed_counts,
            }
        )
        if fill_count <= 0:
            role_record.update(
                {
                    "native_role_capture_status": "NO_FILL_NOT_APPLICABLE",
                    "native_role_hold_reason": "no_observed_fill",
                    "maker_taker_attribution_source": "NO_FILL",
                }
            )
        elif _known_count(observed_counts) >= fill_count:
            role_record.update(
                {
                    "native_role_capture_status": "OBSERVED_PRESERVED",
                    "native_role_hold_reason": "already_has_full_native_role_counts",
                    "maker_taker_attribution_source": "OBSERVED_PFILL_LABEL",
                }
            )
        elif role_source is not None and _known_count(role_source["maker_taker_role_counts"]) >= fill_count:
            evidence = _base_record(run_id, seq, timestamp_ns, "PHASE51Q_FORWARD_NATIVE_ROLE_EVIDENCE")
            seq += 1
            evidence.update(
                {
                    "canonical_group_id": group,
                    "order_key": label.get("order_key"),
                    "venue_id": venue_id,
                    "source_telemetry_sha256": label.get("source_telemetry_sha256"),
                    "fill_count": fill_count,
                    "maker_taker_role_counts": role_source["maker_taker_role_counts"],
                    "maker_taker_attribution_source": role_source["maker_taker_attribution_source"],
                    "source_record_sha256": role_source.get("source_record_sha256"),
                    "native_role_capture_status": "RECOVERED_FORWARD_NATIVE_ROLE",
                }
            )
            native_role_evidence.append(evidence)
            role_record.update(
                {
                    "native_role_capture_status": "RECOVERED_FORWARD_NATIVE_ROLE",
                    "native_role_hold_reason": "explicit_venue_native_source_available",
                    "maker_taker_attribution_source": role_source["maker_taker_attribution_source"],
                }
            )
        elif role_source is not None:
            role_record.update(
                {
                    "native_role_capture_status": "PARTIAL_FORWARD_NATIVE_ROLE_SOURCE",
                    "native_role_hold_reason": "native_role_counts_do_not_cover_fill_count",
                    "maker_taker_attribution_source": role_source["maker_taker_attribution_source"],
                }
            )
        else:
            role_record.update(
                {
                    "native_role_capture_status": "MISSING_FORWARD_NATIVE_ROLE_SOURCE",
                    "native_role_hold_reason": "missing_canonical_group_native_role_source",
                    "maker_taker_attribution_source": "UNRECOVERED",
                }
            )
        role_labels.append(role_record)

        limit_source = limit_by_group.get(group)
        limit_status, limit_reason = _limit_status(venue_id, limit_source)
        limit_record = _base_record(
            run_id,
            seq,
            timestamp_ns,
            "PHASE51Q_NATIVE_LIMIT_PRESSURE_LABEL",
        )
        seq += 1
        limit_record.update(
            {
                "canonical_group_id": group,
                "order_key": label.get("order_key"),
                "venue_id": venue_id,
                "native_limit_pressure_status": limit_status,
                "native_limit_pressure_hold_reason": limit_reason,
            }
        )
        if limit_source is not None:
            for key in (
                "active_order_headroom_account",
                "active_order_headroom_market",
                "sendtx_per_minute_limit",
                "sendtx_per_minute_remaining",
                "rest_requests_per_minute_limit",
                "rest_requests_per_minute_remaining",
                "weighted_requests_per_minute_limit",
                "weighted_requests_per_minute_remaining",
                "native_limit_event_time_status",
                "native_limit_staleness_ms",
                "native_limit_source_sha256",
            ):
                limit_record[key] = limit_source.get(key)
        limit_labels.append(limit_record)

    native_role_evidence_path = out_dir / "native_role_evidence.jsonl"
    role_labels_path = out_dir / "forward_native_role_capture_labels.jsonl"
    limit_labels_path = out_dir / "native_limit_pressure_labels.jsonl"
    summary_path = out_dir / "phase51q_forward_native_evidence_summary.json"
    manifest_path = out_dir / "phase51q_manifest.json"

    _write_jsonl(native_role_evidence_path, native_role_evidence)
    _write_jsonl(role_labels_path, role_labels)
    _write_jsonl(limit_labels_path, limit_labels)

    role_status_counts = _status_counts(role_labels, "native_role_capture_status")
    limit_status_counts = _status_counts(limit_labels, "native_limit_pressure_status")
    recovered_count = role_status_counts.get("RECOVERED_FORWARD_NATIVE_ROLE", 0)
    missing_role_count = role_status_counts.get("MISSING_FORWARD_NATIVE_ROLE_SOURCE", 0)
    partial_role_count = role_status_counts.get("PARTIAL_FORWARD_NATIVE_ROLE_SOURCE", 0)
    missing_limit_count = limit_status_counts.get("MISSING_NATIVE_LIMIT_PRESSURE_SOURCE", 0)
    partial_limit_count = limit_status_counts.get("PARTIAL_NATIVE_LIMIT_PRESSURE_SOURCE", 0)
    gate_reason = (
        "phase51q_forward_native_evidence_complete_nonlive_hold"
        if missing_role_count == 0 and partial_role_count == 0 and missing_limit_count == 0 and partial_limit_count == 0
        else "phase51q_forward_native_evidence_incomplete"
    )

    summary = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "source_observed_pfill_run": str(observed_pfill_run),
        "source_observed_pfill_summary_sha256": _sha256_file(observed_pfill_run / "pfill_outcome_summary.json"),
        "source_observed_pfill_labels_sha256": _sha256_file(observed_pfill_run / "pfill_order_labels.jsonl"),
        "observed_pfill_label_count": len(labels),
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
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
        "native_role_evidence_record_count": len(native_role_evidence),
        "recovered_forward_native_role_count": recovered_count,
        "native_role_capture_status_counts": role_status_counts,
        "native_role_source_counts": _source_counts(native_role_evidence, "maker_taker_attribution_source"),
        "native_limit_pressure_status_counts": limit_status_counts,
        "raw_identifier_redaction_status": "PASS",
        "native_role_source_artifacts": role_source_infos,
        "native_limit_source_artifacts": limit_source_infos,
        "observed_pfill_gate_status": observed_summary.get("gate_status"),
        "observed_pfill_gate_reason": observed_summary.get("gate_reason"),
    }
    _write_json(summary_path, summary)

    artifacts = [
        native_role_evidence_path,
        role_labels_path,
        limit_labels_path,
        summary_path,
    ]
    manifest = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "artifacts": _artifact_infos(out_dir, artifacts),
    }
    _write_json(manifest_path, manifest)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--native-role-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--native-limit-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51q_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_forward_native_evidence(
            observed_pfill_run=args.observed_pfill_run,
            native_role_jsonl=args.native_role_jsonl,
            native_limit_jsonl=args.native_limit_jsonl,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51q_forward_native_evidence_capture: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51q_forward_native_evidence_capture: wrote {out_dir}")
    print("phase51q_forward_native_evidence_capture: status HOLD (forward native evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
