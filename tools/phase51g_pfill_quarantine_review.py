#!/usr/bin/env python3
"""Build the Phase 5.1g P_fill quarantine review evidence pack.

This is an offline evidence gate. It consumes a Phase 5.1f canonical P_fill
pack, preserves every canonical group in a review artifact, and emits a
separate observed-only compatibility pack for diagnostic calibration-readiness
reruns. It does not train a model, submit orders, approve EV admission,
approve live/canary use, or make financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51g_pfill_quarantine_review"

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redacted_compat_source(label: dict[str, Any]) -> dict[str, Any]:
    result = dict(label)
    decision_id = label.get("decision_id")
    for field in RAW_IDENTIFIER_FIELDS:
        result.pop(field, None)
    if decision_id not in (None, ""):
        result["decision_id_present"] = True
        result["decision_id_hash"] = _stable_hash(["decision_id", str(decision_id)])
    else:
        result.setdefault("decision_id_present", False)
        result.setdefault("decision_id_hash", None)
    return result


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


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


def _resolve_run_path(path: Path) -> Path:
    if not path.is_absolute():
        path = ROOT / path
    return path


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _load_hold_summary(path: Path, expected_file: str) -> dict[str, Any]:
    summary_path = path / expected_file
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    return summary


def _empty_counts() -> dict[str, int]:
    return {
        "order_label_count": 0,
        "observed_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "censored_count": 0,
        "train_count": 0,
        "holdout_count": 0,
    }


def _increment_counts(counts: dict[str, int], label: dict[str, Any]) -> None:
    split = str(label.get("order_holdout_split") or "UNKNOWN").upper()
    if split not in {"TRAIN", "HOLDOUT"}:
        raise ValueError(f"unexpected order_holdout_split: {split}")
    outcome_status = str(label.get("outcome_status") or "")
    counts["order_label_count"] += 1
    counts["holdout_count" if split == "HOLDOUT" else "train_count"] += 1
    if outcome_status == "OBSERVED_FILLED":
        if label.get("p_fill_outcome") != 1.0:
            raise ValueError("OBSERVED_FILLED labels must carry p_fill_outcome=1.0")
        counts["observed_count"] += 1
        counts["filled_count"] += 1
    elif outcome_status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
        if label.get("p_fill_outcome") != 0.0:
            raise ValueError("OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL labels must carry p_fill_outcome=0.0")
        counts["observed_count"] += 1
        counts["not_filled_count"] += 1
    elif outcome_status == "CENSORED_OR_UNOBSERVED":
        if label.get("p_fill_outcome") is not None:
            raise ValueError("censored labels must not carry numeric p_fill_outcome")
        counts["censored_count"] += 1
    else:
        raise ValueError(f"unexpected P_fill outcome_status: {outcome_status}")


def _source_status_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        for status, count in (label.get("source_canonical_status_counts") or {}).items():
            counts[str(status)] = counts.get(str(status), 0) + int(count)
    return dict(sorted(counts.items()))


def _exclusion_reason(label: dict[str, Any]) -> str | None:
    if label.get("outcome_status") != "CENSORED_OR_UNOBSERVED":
        return None
    status_counts = label.get("source_canonical_status_counts") or {}
    if status_counts.get("DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW"):
        return "EXCLUDED_DUPLICATE_ALIAS_NO_TERMINAL"
    if status_counts.get("CENSORED_TO_REPLACE_CHAIN_REVIEW"):
        return "EXCLUDED_REPLACE_CHAIN_REVIEW"
    if status_counts.get("CANCEL_ALL_SCOPE_REVIEW"):
        return "EXCLUDED_CANCEL_ALL_SCOPE_REVIEW"
    if status_counts.get("REMAINS_NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW"):
        return "RIGHT_CENSORED_NO_TERMINAL"
    return "EXCLUDED_UNCLASSIFIED_QUARANTINE"


def _review_record(label: dict[str, Any], seq: int, run_id: str, timestamp_ns: int) -> dict[str, Any]:
    outcome_status = str(label.get("outcome_status") or "")
    exclusion_reason = _exclusion_reason(label)
    if outcome_status == "OBSERVED_FILLED":
        review_status = "BINARY_OBSERVED_FILLED_DIAGNOSTIC"
        review_reason = "canonical label has observed fill evidence"
        binary_observed = True
    elif outcome_status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
        review_status = "BINARY_OBSERVED_NOT_FILLED_DIAGNOSTIC"
        review_reason = "canonical label has observed direct terminal evidence"
        binary_observed = True
    else:
        review_status = exclusion_reason or "EXCLUDED_UNCLASSIFIED_QUARANTINE"
        review_reason = "quarantined canonical P_fill group excluded from numeric outcomes"
        binary_observed = False
    return {
        "schema_version": 1,
        "review_seq": seq,
        "label_type": "PHASE51G_PFILL_QUARANTINE_REVIEW_LABEL",
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_model_training": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "admissible_for_model_training": False,
        "source_telemetry_sha256": label.get("source_telemetry_sha256"),
        "source_canonical_run_id": label.get("run_id"),
        "canonical_order_key": label.get("order_key"),
        "canonical_group_id": label.get("canonical_group_id"),
        "venue_id": label.get("venue_id"),
        "side": label.get("side"),
        "order_holdout_split": label.get("order_holdout_split"),
        "source_label_count": label.get("source_label_count"),
        "source_canonical_status_counts": label.get("source_canonical_status_counts"),
        "source_current_status_counts": label.get("source_current_status_counts"),
        "source_old_split_conflict": label.get("source_old_split_conflict"),
        "source_old_split_values": label.get("source_old_split_values"),
        "input_outcome_status": outcome_status,
        "input_p_fill_outcome": label.get("p_fill_outcome"),
        "review_status": review_status,
        "review_reason": review_reason,
        "binary_observed_diagnostic": binary_observed,
        "included_in_observed_only_pack": binary_observed,
        "excluded_from_binary_pack_reason": exclusion_reason,
    }


def _compat_label(label: dict[str, Any], seq: int, run_id: str, timestamp_ns: int) -> dict[str, Any]:
    result = _redacted_compat_source(label)
    result.update({
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "source": "phase51g_observed_only_binary_diagnostic_pack",
        "phase51g_review_status": (
            "BINARY_OBSERVED_FILLED_DIAGNOSTIC"
            if label.get("outcome_status") == "OBSERVED_FILLED"
            else "BINARY_OBSERVED_NOT_FILLED_DIAGNOSTIC"
        ),
        "training_hold_reason": "observed_only_diagnostic_requires_board_review_and_censoring_bias_analysis",
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "admissible_for_model_training": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "no_live_flag": True,
    })
    return result


def build_quarantine_review(
    *,
    canonical_pfill_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51G-PFILL-QUARANTINE-REVIEW-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    canonical_pfill_run = _resolve_run_path(canonical_pfill_run)
    input_summary = _load_hold_summary(canonical_pfill_run, "pfill_outcome_summary.json")
    labels_path = canonical_pfill_run / "canonical_pfill_order_labels.jsonl"
    if not labels_path.exists():
        labels_path = canonical_pfill_run / "pfill_order_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        labels.append(label)

    input_counts = _empty_counts()
    for label in labels:
        _increment_counts(input_counts, label)
    expected = {
        "order_label_count": int(input_summary.get("order_label_count") or 0),
        "filled_count": int(input_summary.get("filled_count") or 0),
        "not_filled_count": int(input_summary.get("not_filled_count") or 0),
        "censored_count": int(input_summary.get("censored_count") or 0),
    }
    actual = {key: input_counts[key] for key in expected}
    if actual != expected:
        raise ValueError(f"canonical P_fill summary counts do not reconcile: {actual} != {expected}")

    review_records = [
        _review_record(label, seq, run_id, timestamp_ns)
        for seq, label in enumerate(labels, start=1)
    ]
    observed_labels = [
        label
        for label in labels
        if label.get("outcome_status") in {"OBSERVED_FILLED", "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL"}
    ]
    compat_run_id = f"{run_id}-OBSERVED-ONLY-COMPAT"
    compat_labels = [
        _compat_label(label, seq, compat_run_id, timestamp_ns)
        for seq, label in enumerate(observed_labels, start=1)
    ]
    compat_counts = _empty_counts()
    for label in compat_labels:
        _increment_counts(compat_counts, label)

    exclusion_reason_counts: dict[str, int] = {}
    venue_quarantine_counts: dict[str, int] = {}
    split_conflict_counts = {"true": 0, "false": 0}
    for record in review_records:
        reason = record.get("excluded_from_binary_pack_reason")
        if reason:
            exclusion_reason_counts[str(reason)] = exclusion_reason_counts.get(str(reason), 0) + 1
            venue = str(record.get("venue_id") or "UNKNOWN")
            venue_quarantine_counts[venue] = venue_quarantine_counts.get(venue, 0) + 1
        split_conflict_counts["true" if record.get("source_old_split_conflict") else "false"] += 1

    source_reconciliation = [
        {
            "canonical_order_key": record.get("canonical_order_key"),
            "canonical_group_id": record.get("canonical_group_id"),
            "source_telemetry_sha256": record.get("source_telemetry_sha256"),
            "venue_id": record.get("venue_id"),
            "review_status": record.get("review_status"),
            "included_in_observed_only_pack": record.get("included_in_observed_only_pack"),
            "excluded_from_binary_pack_reason": record.get("excluded_from_binary_pack_reason"),
            "source_label_count": record.get("source_label_count"),
            "source_canonical_status_counts": record.get("source_canonical_status_counts"),
            "source_old_split_conflict": record.get("source_old_split_conflict"),
        }
        for record in review_records
    ]

    review_path = out_dir / "quarantine_review_labels.jsonl"
    observed_labels_path = out_dir / "binary_observed_pfill_order_labels.jsonl"
    source_manifest_path = out_dir / "source_reconciliation_manifest.jsonl"
    summary_path = out_dir / "quarantine_review_summary.json"
    compat_dir = out_dir / "observed_only_pfill_outcome"
    compat_labels_path = compat_dir / "pfill_order_labels.jsonl"
    compat_summary_path = compat_dir / "pfill_outcome_summary.json"

    _write_jsonl(review_path, review_records)
    _write_jsonl(observed_labels_path, compat_labels)
    _write_jsonl(source_manifest_path, source_reconciliation)
    _write_jsonl(compat_labels_path, compat_labels)

    source_telemetry_sha256_list = sorted({
        str(label.get("source_telemetry_sha256") or "")
        for label in labels
        if label.get("source_telemetry_sha256")
    })
    summary = {
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51g_quarantine_review_observed_only_diagnostic_pack",
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
        "canonical_pfill_run": str(canonical_pfill_run),
        "canonical_pfill_summary_sha256": _sha256_file(canonical_pfill_run / "pfill_outcome_summary.json"),
        "canonical_pfill_labels_sha256": _sha256_file(labels_path),
        "source_telemetry_sha256": _stable_hash(source_telemetry_sha256_list),
        "source_telemetry_sha256_list": source_telemetry_sha256_list,
        "source_canonical_status_counts": _source_status_counts(labels),
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "venue_quarantine_counts": dict(sorted(venue_quarantine_counts.items())),
        "source_old_split_conflict_counts": split_conflict_counts,
        "observed_only_pack_run_id": compat_run_id,
        "observed_only_pack_path": str(compat_dir),
        "observed_only_pack_warning": "diagnostic only; excludes censored groups and may carry selection bias",
        **input_counts,
    }
    _write_json(summary_path, summary)

    compat_summary = {
        "run_id": compat_run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51g_observed_only_binary_diagnostic_requires_board_review",
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "no_live_flag": True,
        "canonical_pfill_run": str(canonical_pfill_run),
        "phase51g_quarantine_review_run": str(out_dir),
        "source_telemetry_sha256": summary["source_telemetry_sha256"],
        "source_telemetry_sha256_list": source_telemetry_sha256_list,
        "excluded_quarantine_count": input_counts["censored_count"],
        "excluded_quarantine_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "observed_only_pack_warning": "diagnostic only; excluded censored groups are not negative outcomes",
        **{key: compat_counts[key] for key in ("order_label_count", "filled_count", "not_filled_count", "censored_count", "train_count", "holdout_count")},
    }
    _write_json(compat_summary_path, compat_summary)

    artifact_paths = [
        review_path,
        observed_labels_path,
        source_manifest_path,
        summary_path,
        compat_labels_path,
        compat_summary_path,
    ]
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
    parser.add_argument("--canonical-pfill-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_quarantine_review(
            canonical_pfill_run=args.canonical_pfill_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51g_pfill_quarantine_review: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51g_pfill_quarantine_review: wrote {out_dir}")
    print("phase51g_pfill_quarantine_review: status HOLD (quarantine review only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
