#!/usr/bin/env python3
"""Build Phase 5.1u forward native capture target manifests.

This HOLD-only gate consumes canonical observed P_fill labels and emits the
exact redacted targets still needed for forward venue-native maker/taker role
capture and Lighter event-time native-limit pressure capture. It performs no
network access, reads no secrets, submits no orders, and infers no economics.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51u_forward_capture_target_manifest"
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
    "id",
    "oid",
    "cloid",
    "tid",
}
ROLE_VALUES = {"MAKER", "TAKER", "UNKNOWN"}
VENUE_ROLE_REQUIREMENTS = {
    "lighter": {
        "required_native_role_source": "LIGHTER_TRADES_JSON",
        "required_native_role_fields": [
            "account_index",
            "is_maker_ask",
            "ask_account_id",
            "bid_account_id",
        ],
    },
    "hyperliquid": {
        "required_native_role_source": "HYPERLIQUID_CROSSED",
        "required_native_role_fields": ["crossed"],
    },
    "paradex": {
        "required_native_role_source": "PARADEX_LIQUIDITY",
        "required_native_role_fields": ["liquidity"],
    },
    "aster": {
        "required_native_role_source": "ASTER_ORDER_TRADE_UPDATE_M",
        "required_native_role_fields": [
            "e=ORDER_TRADE_UPDATE or equivalent order-trade-update envelope",
            "o.m or m",
            "positive o.l or lastFilledQty",
        ],
    },
    "extended": {
        "required_native_role_source": "EXTENDED_ISTAKER",
        "required_native_role_fields": ["isTaker or is_taker"],
    },
}
LIGHTER_NATIVE_LIMIT_REQUIRED_FIELDS = [
    "active_order_headroom_account",
    "active_order_headroom_market",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "rest_requests_per_minute_limit/rest_requests_per_minute_remaining or "
    "weighted_requests_per_minute_limit/weighted_requests_per_minute_remaining",
    "native_limit_event_time_status",
]
LIGHTER_NATIVE_LIMIT_ACCEPTED_ALIGNMENT = [
    "EVENT_TIME_ALIGNED",
    "SNAPSHOT_AT_DECISION_TIME",
    "OBSERVED_AT_DECISION_TIME",
]
JOIN_PATHS = [
    "canonical_group_id on the sanitized source row",
    "order_key on the sanitized source row",
    "Phase 5.1t/5.1s source-link sidecar keyed by redacted source-record hash",
]


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
            _check_output_safe(record, path)
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


def _role_counts(value: Any) -> dict[str, int]:
    counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
    if isinstance(value, dict):
        for key in ROLE_VALUES:
            counts[key] = max(0, _safe_int(value.get(key)) or 0)
    return counts


def _known_role_count(counts: dict[str, int]) -> int:
    return int(counts.get("MAKER") or 0) + int(counts.get("TAKER") or 0)


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _venue_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        venue = str(record.get("venue_id") or "unknown")
        counts[venue] = counts.get(venue, 0) + 1
    return dict(sorted(counts.items()))


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")
    raw_fields = RAW_IDENTIFIER_FIELDS & set(record)
    if raw_fields:
        raise ValueError(f"{path} has raw {label} identifier fields: {sorted(raw_fields)}")


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe(record, path, label="output")


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
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
    }


def _load_observed_pfill(observed_pfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = observed_pfill_run / "pfill_outcome_summary.json"
    labels_path = observed_pfill_run / "pfill_order_labels.jsonl"
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
    expected = _safe_int(summary.get("order_label_count"))
    if expected is not None and len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary order_label_count {expected}")
    return summary, labels


def _common_target_fields(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_group_id": str(label.get("canonical_group_id") or ""),
        "order_key": str(label.get("order_key") or ""),
        "venue_id": str(label.get("venue_id") or "unknown").lower(),
        "side": label.get("side"),
        "price": label.get("price"),
        "size": label.get("size"),
        "order_source_t": label.get("order_source_t"),
        "order_source_line": label.get("order_source_line"),
        "source_telemetry_sha256": label.get("source_telemetry_sha256"),
        "source_order_key_count": len(label.get("source_order_keys") or []),
        "accepted_join_paths": JOIN_PATHS,
        "downstream_staging_tool": "tools/phase51s_local_native_source_acquisition.py",
        "downstream_source_acquisition_tool": "tools/phase51r_forward_native_source_acquisition.py",
        "source_link_builder_tool": "tools/phase51t_source_link_sidecar_builder.py",
    }


def _role_target(
    label: dict[str, Any],
    run_id: str,
    seq: int,
    timestamp_ns: int,
) -> dict[str, Any]:
    venue = str(label.get("venue_id") or "unknown").lower()
    requirements = VENUE_ROLE_REQUIREMENTS.get(venue, {
        "required_native_role_source": "VENUE_NATIVE_FILL_FIELD",
        "required_native_role_fields": ["native_role or native_liquidity_role with native provenance"],
    })
    fill_count = _safe_int(label.get("fill_count")) or 0
    role_counts = _role_counts(label.get("maker_taker_role_counts"))
    known_count = _known_role_count(role_counts)
    record = _base_record(run_id, seq, timestamp_ns, "PHASE51U_NATIVE_ROLE_CAPTURE_TARGET")
    record.update(_common_target_fields(label))
    record.update(
        {
            "target_reason": "filled_order_missing_complete_native_role",
            "fill_count": fill_count,
            "known_native_role_count": known_count,
            "missing_native_role_count": max(0, fill_count - known_count),
            "existing_maker_taker_role_counts": role_counts,
            "required_native_role_source": requirements["required_native_role_source"],
            "required_native_role_fields": requirements["required_native_role_fields"],
            "role_inference_allowed": False,
            "post_only_or_strategy_intent_allowed_as_role_source": False,
            "clears_phase51_blockers": False,
        }
    )
    return record


def _lighter_limit_target(
    label: dict[str, Any],
    run_id: str,
    seq: int,
    timestamp_ns: int,
) -> dict[str, Any]:
    record = _base_record(run_id, seq, timestamp_ns, "PHASE51U_LIGHTER_NATIVE_LIMIT_CAPTURE_TARGET")
    record.update(_common_target_fields(label))
    record.update(
        {
            "target_reason": "lighter_event_time_native_limit_pressure_required",
            "required_native_limit_fields": LIGHTER_NATIVE_LIMIT_REQUIRED_FIELDS,
            "accepted_native_limit_event_time_status": LIGHTER_NATIVE_LIMIT_ACCEPTED_ALIGNMENT,
            "requires_active_order_pressure": True,
            "requires_sendtx_pressure": True,
            "requires_rest_or_weighted_request_pressure": True,
            "doc_caps_alone_sufficient": False,
            "current_snapshot_without_event_time_alignment_sufficient": False,
            "clears_phase51_blockers": False,
        }
    )
    return record


def _capture_template(
    run_id: str,
    observed_pfill_run: Path,
    role_targets: list[dict[str, Any]],
    limit_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    source_entries = []
    for venue, requirements in sorted(VENUE_ROLE_REQUIREMENTS.items()):
        source_entries.append(
            {
                "source_id": f"{venue}_forward_native_role_snapshot",
                "venue_id": venue,
                "path": f"<local-read-only-{venue}-native-role-snapshot.jsonl>",
                "required_native_role_source": requirements["required_native_role_source"],
                "required_native_role_fields": requirements["required_native_role_fields"],
                "must_include_one_join_path": JOIN_PATHS,
                "live_orders_allowed": False,
            }
        )
    source_entries.append(
        {
            "source_id": "lighter_forward_native_limit_pressure_snapshot",
            "venue_id": "lighter",
            "path": "<local-read-only-lighter-native-limit-pressure-snapshot.jsonl>",
            "required_native_limit_fields": LIGHTER_NATIVE_LIMIT_REQUIRED_FIELDS,
            "accepted_native_limit_event_time_status": LIGHTER_NATIVE_LIMIT_ACCEPTED_ALIGNMENT,
            "must_include_one_join_path": JOIN_PATHS,
            "live_orders_allowed": False,
        }
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "observed_pfill_run": str(observed_pfill_run),
        "purpose": "template for assembling fresh local source snapshots before Phase 5.1s",
        "native_role_capture_target_count": len(role_targets),
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "sources": source_entries,
        "source_links": [
            {
                "source_link_id": "optional_phase51t_or_external_redacted_source_links",
                "path": "<local-redacted-source-links.sanitized.jsonl>",
                "allowed_fields": [
                    "phase51s_source_record_sha256/source_record_sha256/redacted_source_record_sha256",
                    "canonical_group_id or order_key",
                    "false safety authorization flags",
                ],
                "role_or_limit_inference_allowed": False,
            }
        ],
        "downstream_chain": [
            "tools/phase51s_local_native_source_acquisition.py",
            "tools/phase51r_forward_native_source_acquisition.py",
            "tools/phase51q_forward_native_evidence_capture.py",
            "tools/phase51n_maker_taker_attribution_recovery.py",
            "tools/phase51h_observed_pfill_feature_audit.py",
            "tools/phase51i_pfill_feature_matrix_admissibility.py",
        ],
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }


def build_forward_capture_target_manifest(
    *,
    observed_pfill_run: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    observed_pfill_run = _resolve_path(observed_pfill_run)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_summary, labels = _load_observed_pfill(observed_pfill_run)
    role_targets: list[dict[str, Any]] = []
    limit_targets: list[dict[str, Any]] = []
    seq = 0

    for label in labels:
        fill_count = _safe_int(label.get("fill_count")) or 0
        role_counts = _role_counts(label.get("maker_taker_role_counts"))
        if fill_count > _known_role_count(role_counts):
            role_targets.append(_role_target(label, run_id, seq, timestamp_ns))
            seq += 1
        if str(label.get("venue_id") or "").lower() == "lighter":
            limit_targets.append(_lighter_limit_target(label, run_id, seq, timestamp_ns))
            seq += 1

    role_targets.sort(key=lambda item: (str(item["venue_id"]), str(item["canonical_group_id"])))
    limit_targets.sort(key=lambda item: str(item["canonical_group_id"]))

    role_path = out_dir / "native_role_capture_targets.jsonl"
    limit_path = out_dir / "lighter_native_limit_capture_targets.jsonl"
    template_path = out_dir / "capture_bundle_manifest_template.json"
    summary_path = out_dir / "phase51u_forward_capture_target_manifest_summary.json"
    manifest_path = out_dir / "phase51u_manifest.json"
    _write_jsonl(role_path, role_targets)
    _write_jsonl(limit_path, limit_targets)
    _write_json(template_path, _capture_template(run_id, observed_pfill_run, role_targets, limit_targets))

    summary = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51u_forward_capture_targets_emitted_nonlive_hold",
        "observed_pfill_run": str(observed_pfill_run),
        "observed_pfill_summary_sha256": _sha256_file(observed_pfill_run / "pfill_outcome_summary.json"),
        "observed_pfill_labels_sha256": _sha256_file(observed_pfill_run / "pfill_order_labels.jsonl"),
        "observed_pfill_label_count": len(labels),
        "observed_pfill_summary_order_label_count": _safe_int(observed_summary.get("order_label_count")),
        "native_role_capture_target_count": len(role_targets),
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "native_role_capture_target_counts_by_venue": _venue_counts(role_targets),
        "lighter_native_limit_capture_target_counts_by_venue": _venue_counts(limit_targets),
        "native_role_required_source_counts": _status_counts(role_targets, "required_native_role_source"),
        "required_next_chain": [
            "phase51s",
            "phase51r",
            "phase51q",
            "phase51n",
            "phase51h",
            "phase51i",
        ],
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

    artifacts = [role_path, limit_path, template_path, summary_path]
    manifest = {
        "schema_version": 1,
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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51u_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_forward_capture_target_manifest(
            observed_pfill_run=args.observed_pfill_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51u_forward_capture_target_manifest: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51u_forward_capture_target_manifest: wrote {out_dir}")
    print("phase51u_forward_capture_target_manifest: status HOLD (capture targets only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
