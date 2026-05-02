#!/usr/bin/env python3
"""Build a Phase 5.1c P_fill calibration-readiness evidence pack.

This is an offline evidence gate. It consumes Phase 5.1c order-level P_fill
outcome labels, aggregates immutable train/holdout coverage, and computes
binomial confidence intervals for observed non-censored outcomes. It does not
train a model, tune a strategy, submit orders, approve live/canary use, or make
financial claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_pfill_calibration_readiness"
DEFAULT_BUCKET_FIELDS = ("venue_id", "side")
WILSON_Z_95 = 1.959963984540054
UNSAFE_TRUE_FLAGS = {
    "approved_for_model_training",
    "approved_for_live",
    "approved_for_canary",
    "approved_for_capital_escalation",
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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"bid", "buy"}:
        return "BID"
    if side in {"ask", "sell"}:
        return "ASK"
    return "UNKNOWN"


def _bucket_value(label: dict[str, Any], field: str) -> str:
    if field == "side":
        return _canonical_side(label.get(field))
    value = label.get(field)
    if value is None or value == "":
        return "UNKNOWN"
    return str(value)


def _wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> dict[str, float | None]:
    if total <= 0:
        return {"rate": None, "ci_low": None, "ci_high": None}
    p_hat = successes / total
    denom = 1.0 + z * z / total
    centre = p_hat + z * z / (2.0 * total)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
    return {
        "rate": p_hat,
        "ci_low": max(0.0, (centre - margin) / denom),
        "ci_high": min(1.0, (centre + margin) / denom),
    }


def _empty_counts() -> dict[str, int]:
    return {
        "order_label_count": 0,
        "observed_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "censored_count": 0,
        "missing_observed_horizon_count": 0,
        "train_observed_count": 0,
        "train_filled_count": 0,
        "train_not_filled_count": 0,
        "train_censored_count": 0,
        "holdout_observed_count": 0,
        "holdout_filled_count": 0,
        "holdout_not_filled_count": 0,
        "holdout_censored_count": 0,
    }


def _increment_counts(counts: dict[str, int], label: dict[str, Any]) -> None:
    outcome_status = str(label.get("outcome_status") or "")
    outcome = _safe_float(label.get("p_fill_outcome"))
    if outcome_status not in {
        "OBSERVED_FILLED",
        "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
        "CENSORED_OR_UNOBSERVED",
    }:
        raise ValueError(f"unexpected P_fill outcome_status: {outcome_status}")
    if outcome_status == "CENSORED_OR_UNOBSERVED" and outcome is not None:
        raise ValueError("censored P_fill labels must not carry numeric outcomes")
    if outcome_status != "CENSORED_OR_UNOBSERVED" and outcome is None:
        raise ValueError("observed P_fill labels must carry numeric outcomes")
    if outcome_status == "OBSERVED_FILLED" and outcome != 1.0:
        raise ValueError("OBSERVED_FILLED P_fill labels must carry p_fill_outcome=1.0")
    if outcome_status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL" and outcome != 0.0:
        raise ValueError("OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL labels must carry p_fill_outcome=0.0")
    split = str(label.get("order_holdout_split") or "UNKNOWN").upper()
    if split not in {"TRAIN", "HOLDOUT"}:
        raise ValueError(f"unexpected order_holdout_split: {split}")
    split_prefix = "holdout" if split == "HOLDOUT" else "train"
    counts["order_label_count"] += 1
    if label.get("observed_horizon_source_ticks") is None:
        counts["missing_observed_horizon_count"] += 1
    if outcome is None:
        counts["censored_count"] += 1
        counts[f"{split_prefix}_censored_count"] += 1
        return
    counts["observed_count"] += 1
    counts[f"{split_prefix}_observed_count"] += 1
    if outcome > 0.0:
        counts["filled_count"] += 1
        counts[f"{split_prefix}_filled_count"] += 1
    else:
        counts["not_filled_count"] += 1
        counts[f"{split_prefix}_not_filled_count"] += 1


def _base_bucket_record(
    *,
    run_id: str,
    seq: int,
    timestamp_ns: int,
    bucket_id: str,
    bucket_dimensions: dict[str, str],
    counts: dict[str, int],
    min_observed_per_bucket: int,
    min_holdout_observed_per_bucket: int,
) -> dict[str, Any]:
    train_ci = _wilson_interval(counts["train_filled_count"], counts["train_observed_count"])
    holdout_ci = _wilson_interval(counts["holdout_filled_count"], counts["holdout_observed_count"])
    all_ci = _wilson_interval(counts["filled_count"], counts["observed_count"])
    sparse = (
        counts["observed_count"] < min_observed_per_bucket
        or counts["holdout_observed_count"] < min_holdout_observed_per_bucket
    )
    censored = counts["censored_count"] > 0
    reasons: list[str] = []
    if sparse:
        reasons.append("sparse_pfill_calibration_bucket")
    if censored:
        reasons.append("censored_orders_present")
    reasons.append("requires_feature_rich_pfill_model_and_board_review")
    return {
        "schema_version": 1,
        "label_type": "PFILL_CALIBRATION_BUCKET_READINESS",
        "bucket_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "bucket_id": bucket_id,
        "bucket_dimensions": bucket_dimensions,
        "gate_status": "HOLD",
        "gate_reasons": reasons,
        "min_observed_per_bucket": min_observed_per_bucket,
        "min_holdout_observed_per_bucket": min_holdout_observed_per_bucket,
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
        "censored_rate": (
            counts["censored_count"] / counts["order_label_count"]
            if counts["order_label_count"] > 0
            else None
        ),
        "train_censored_rate": (
            counts["train_censored_count"] / (counts["train_censored_count"] + counts["train_observed_count"])
            if counts["train_censored_count"] + counts["train_observed_count"] > 0
            else None
        ),
        "holdout_censored_rate": (
            counts["holdout_censored_count"] / (counts["holdout_censored_count"] + counts["holdout_observed_count"])
            if counts["holdout_censored_count"] + counts["holdout_observed_count"] > 0
            else None
        ),
        "observed_fill_rate": all_ci["rate"],
        "observed_fill_rate_ci_low_95": all_ci["ci_low"],
        "observed_fill_rate_ci_high_95": all_ci["ci_high"],
        "train_fill_rate": train_ci["rate"],
        "train_fill_rate_ci_low_95": train_ci["ci_low"],
        "train_fill_rate_ci_high_95": train_ci["ci_high"],
        "holdout_fill_rate": holdout_ci["rate"],
        "holdout_fill_rate_ci_low_95": holdout_ci["ci_low"],
        "holdout_fill_rate_ci_high_95": holdout_ci["ci_high"],
    }


def _load_outcome_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_json(path / "pfill_outcome_summary.json")
    for flag in UNSAFE_TRUE_FLAGS:
        if summary.get(flag) is True:
            raise ValueError(f"{path} has unsafe summary flag {flag}=true")
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(path / "pfill_order_labels.jsonl"):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        for flag in UNSAFE_TRUE_FLAGS:
            if label.get(flag) is True:
                raise ValueError(f"{path} has unsafe label flag {flag}=true")
        labels.append(label)
    return summary, labels


def _split_manifest_record(label: dict[str, Any], run_path: Path) -> dict[str, Any]:
    return {
        "order_key": str(label.get("order_key")),
        "order_holdout_split": str(label.get("order_holdout_split") or "UNKNOWN").upper(),
        "source_run_path": str(run_path),
        "source_run_id": label.get("run_id"),
        "source_telemetry_sha256": label.get("source_telemetry_sha256"),
        "venue_id": label.get("venue_id"),
        "side": _canonical_side(label.get("side")),
        "outcome_status": label.get("outcome_status"),
    }


def _gate_reason(total_counts: dict[str, int], bucket_records: list[dict[str, Any]]) -> str:
    if total_counts["order_label_count"] == 0:
        return "pfill_calibration_missing_order_labels"
    if total_counts["observed_count"] == 0:
        return "pfill_calibration_missing_observed_outcomes"
    if total_counts["holdout_observed_count"] == 0:
        return "pfill_calibration_missing_holdout_outcomes"
    if total_counts["censored_count"] > 0:
        return "pfill_calibration_contains_censored_orders"
    if any("sparse_pfill_calibration_bucket" in record["gate_reasons"] for record in bucket_records):
        return "pfill_calibration_sparse_buckets"
    return "pfill_calibration_requires_feature_rich_model_and_board_review"


def build_pfill_calibration_readiness(
    *,
    pfill_outcome_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    bucket_fields: tuple[str, ...],
    min_observed_per_bucket: int,
    min_holdout_observed_per_bucket: int,
) -> Path:
    if not pfill_outcome_runs:
        raise ValueError("at least one --pfill-outcome-run is required")
    run_id = run_id or f"PHASE51C-PFILL-CALIBRATION-READINESS-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    input_summaries: list[dict[str, Any]] = []
    bucket_counts: dict[str, dict[str, Any]] = {}
    total_counts = _empty_counts()
    source_telemetry_sha256: set[str] = set()
    split_manifest_by_order: dict[str, dict[str, Any]] = {}
    terminal_action_counts: dict[str, int] = {}
    for run_path in pfill_outcome_runs:
        summary, labels = _load_outcome_run(run_path)
        input_summaries.append({
            "run_path": str(run_path),
            "run_id": summary.get("run_id"),
            "gate_status": summary.get("gate_status"),
            "gate_reason": summary.get("gate_reason"),
            "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
            "pfill_outcome_summary_sha256": _sha256_file(run_path / "pfill_outcome_summary.json"),
            "pfill_order_labels_sha256": _sha256_file(run_path / "pfill_order_labels.jsonl"),
        })
        if summary.get("source_telemetry_sha256"):
            source_telemetry_sha256.add(str(summary.get("source_telemetry_sha256")))
        for label in labels:
            order_key = label.get("order_key")
            if not order_key:
                raise ValueError(f"missing order_key in {run_path}")
            split_record = _split_manifest_record(label, run_path)
            existing_split = split_manifest_by_order.get(str(order_key))
            if existing_split and existing_split["order_holdout_split"] != split_record["order_holdout_split"]:
                raise ValueError(f"conflicting order_holdout_split for order_key {order_key}")
            split_manifest_by_order.setdefault(str(order_key), split_record)
            terminal_action = str(label.get("terminal_action_first") or "NONE")
            terminal_action_counts[terminal_action] = terminal_action_counts.get(terminal_action, 0) + 1
            dimensions = {field: _bucket_value(label, field) for field in bucket_fields}
            bucket_id = _stable_hash(dimensions)
            if bucket_id not in bucket_counts:
                bucket_counts[bucket_id] = {
                    "dimensions": dimensions,
                    "counts": _empty_counts(),
                }
            _increment_counts(bucket_counts[bucket_id]["counts"], label)
            _increment_counts(total_counts, label)

    bucket_records: list[dict[str, Any]] = []
    global_record = _base_bucket_record(
        run_id=run_id,
        seq=1,
        timestamp_ns=timestamp_ns,
        bucket_id="GLOBAL",
        bucket_dimensions={"scope": "GLOBAL"},
        counts=total_counts,
        min_observed_per_bucket=min_observed_per_bucket,
        min_holdout_observed_per_bucket=min_holdout_observed_per_bucket,
    )
    bucket_records.append(global_record)
    for seq, (bucket_id, payload) in enumerate(sorted(bucket_counts.items()), start=2):
        bucket_records.append(_base_bucket_record(
            run_id=run_id,
            seq=seq,
            timestamp_ns=timestamp_ns,
            bucket_id=bucket_id,
            bucket_dimensions=payload["dimensions"],
            counts=payload["counts"],
            min_observed_per_bucket=min_observed_per_bucket,
            min_holdout_observed_per_bucket=min_holdout_observed_per_bucket,
        ))

    gate_reason = _gate_reason(total_counts, bucket_records)
    buckets_path = out_dir / "pfill_calibration_buckets.jsonl"
    split_manifest_path = out_dir / "pfill_order_split_manifest.jsonl"
    summary_path = out_dir / "pfill_calibration_readiness_summary.json"
    _write_jsonl(buckets_path, bucket_records)
    split_manifest = [split_manifest_by_order[key] for key in sorted(split_manifest_by_order)]
    _write_jsonl(split_manifest_path, split_manifest)
    summary = {
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
        "bucket_fields": list(bucket_fields),
        "bucket_count": len(bucket_records),
        "source_telemetry_sha256_list": sorted(source_telemetry_sha256),
        "input_pfill_outcome_runs": input_summaries,
        "min_observed_per_bucket": min_observed_per_bucket,
        "min_holdout_observed_per_bucket": min_holdout_observed_per_bucket,
        "terminal_action_counts": terminal_action_counts,
        "order_split_manifest_count": len(split_manifest),
        "order_split_manifest_sha256": _sha256_file(split_manifest_path),
        **total_counts,
        "censored_rate": (
            total_counts["censored_count"] / total_counts["order_label_count"]
            if total_counts["order_label_count"] > 0
            else None
        ),
        "train_censored_rate": (
            total_counts["train_censored_count"] / (total_counts["train_censored_count"] + total_counts["train_observed_count"])
            if total_counts["train_censored_count"] + total_counts["train_observed_count"] > 0
            else None
        ),
        "holdout_censored_rate": (
            total_counts["holdout_censored_count"] / (total_counts["holdout_censored_count"] + total_counts["holdout_observed_count"])
            if total_counts["holdout_censored_count"] + total_counts["holdout_observed_count"] > 0
            else None
        ),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [buckets_path, split_manifest_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [buckets_path, split_manifest_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfill-outcome-run", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--bucket-field", action="append", default=None)
    parser.add_argument("--min-observed-per-bucket", type=int, default=200)
    parser.add_argument("--min-holdout-observed-per-bucket", type=int, default=50)
    args = parser.parse_args()
    bucket_fields = tuple(args.bucket_field or DEFAULT_BUCKET_FIELDS)
    try:
        out_dir = build_pfill_calibration_readiness(
            pfill_outcome_runs=args.pfill_outcome_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            bucket_fields=bucket_fields,
            min_observed_per_bucket=args.min_observed_per_bucket,
            min_holdout_observed_per_bucket=args.min_holdout_observed_per_bucket,
        )
    except Exception as exc:
        print(f"phase51c_pfill_calibration_readiness: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_pfill_calibration_readiness: wrote {out_dir}")
    print("phase51c_pfill_calibration_readiness: status HOLD (calibration readiness only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
