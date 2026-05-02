#!/usr/bin/env python3
"""Build a Phase 5.1c markout/adverse-selection readiness evidence pack.

This is an offline evidence gate. It consumes observed fill/markout labels plus
deterministic join/holdout labels, preserves the existing fill split, and emits
descriptive bucket statistics for markout calibration readiness. It does not
train a model, tune a strategy, submit orders, approve live/canary use, approve
EV admission, or make financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_markout_calibration_readiness"
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


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric markout_pnl, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"expected finite markout_pnl, got {value!r}")
    return result


def _safe_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected integer {field}, got {value!r}") from exc


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
        value = record.get(field)
        if field == "side":
            key = _canonical_side(value)
        elif value is None or value == "":
            key = "UNKNOWN"
        else:
            key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fill_count_map(split_records: dict[str, dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in split_records.values():
        value = record.get(field)
        if field == "side":
            key = _canonical_side(value)
        elif value is None or value == "":
            key = "UNKNOWN"
        else:
            key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _fill_list_value_count_map(split_records: dict[str, dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in split_records.values():
        values = record.get(field)
        if not isinstance(values, list) or not values:
            values = ["UNKNOWN"]
        for value in values:
            key = "UNKNOWN" if value is None or value == "" else str(value)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _validate_hold_summary(summary: dict[str, Any], path: Path, *, expected_source_sha: str | None = None) -> str:
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{path} must have gate_status=HOLD")
    baseline_commit = summary.get("baseline_commit")
    if baseline_commit != BASELINE_COMMIT:
        raise ValueError(f"{path} must use baseline_commit={BASELINE_COMMIT}")
    for flag in UNSAFE_TRUE_FLAGS:
        if summary.get(flag) is True:
            raise ValueError(f"{path} has unsafe summary flag {flag}=true")
    source_sha = summary.get("source_telemetry_sha256")
    if not source_sha:
        raise ValueError(f"{path} missing source_telemetry_sha256")
    if expected_source_sha and source_sha != expected_source_sha:
        raise ValueError(f"{path} source_telemetry_sha256 does not match paired run")
    return str(source_sha)


def _validate_label_flags(record: dict[str, Any], source_path: Path) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{source_path} has unsafe label flag {flag}=true")


def _load_join_run(join_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = join_run / "join_holdout_summary.json"
    summary = _load_json(summary_path)
    _validate_hold_summary(summary, summary_path)
    joined_by_fill: dict[str, dict[str, Any]] = {}
    labels_path = join_run / "joined_labels.jsonl"
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "DETERMINISTIC_JOIN_LABEL":
            continue
        _validate_label_flags(label, labels_path)
        fill_id = label.get("fill_id")
        if not fill_id:
            raise ValueError(f"{labels_path} has join label without fill_id")
        fill_id_s = str(fill_id)
        if fill_id_s in joined_by_fill:
            raise ValueError(f"{labels_path} has duplicate DETERMINISTIC_JOIN_LABEL for fill_id {fill_id_s}")
        split = str(label.get("holdout_split") or "UNKNOWN").upper()
        if split not in {"TRAIN", "HOLDOUT"}:
            raise ValueError(f"{labels_path} has invalid holdout_split={split!r} for fill_id {fill_id_s}")
        joined_by_fill[fill_id_s] = label
    if not joined_by_fill:
        raise ValueError(f"{labels_path} contains no DETERMINISTIC_JOIN_LABEL records")
    return summary, joined_by_fill


def _load_observed_markouts(
    observed_run: Path,
    joined_by_fill: dict[str, dict[str, Any]],
    *,
    expected_source_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary_path = observed_run / "observed_label_summary.json"
    summary = _load_json(summary_path)
    source_sha = _validate_hold_summary(summary, summary_path, expected_source_sha=expected_source_sha)
    markouts: list[dict[str, Any]] = []
    split_manifest: dict[str, dict[str, Any]] = {}
    labels_path = observed_run / "labels.jsonl"
    markout_seen: set[tuple[str, int, int]] = set()
    for line_no, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "OBSERVED_MARKOUT_LABEL":
            continue
        _validate_label_flags(label, labels_path)
        fill_id = label.get("fill_id")
        if not fill_id:
            raise ValueError(f"{labels_path}:{line_no} has markout label without fill_id")
        fill_id_s = str(fill_id)
        join = joined_by_fill.get(fill_id_s)
        if join is None:
            raise ValueError(f"{labels_path}:{line_no} markout fill_id {fill_id_s} has no deterministic join label")
        horizon_ms = _safe_int(label.get("markout_horizon_ms"), field="markout_horizon_ms")
        fill_time_ms = _safe_int(label.get("fill_time_ms"), field="fill_time_ms")
        dedupe_key = (fill_id_s, horizon_ms, fill_time_ms)
        if dedupe_key in markout_seen:
            raise ValueError(f"{labels_path}:{line_no} duplicate markout label for fill_id {fill_id_s} horizon {horizon_ms}")
        markout_seen.add(dedupe_key)
        split = str(join.get("holdout_split") or "UNKNOWN").upper()
        side = _canonical_side(label.get("side") or join.get("side"))
        venue_id = str(label.get("venue_id") or join.get("venue_id") or "UNKNOWN")
        markout = {
            "source_telemetry_sha256": source_sha,
            "fill_id": fill_id_s,
            "holdout_split": split,
            "markout_horizon_ms": horizon_ms,
            "markout_pnl": _safe_float(label.get("markout_pnl")),
            "venue_id": venue_id,
            "side": side,
            "maker_taker_role": str(join.get("maker_taker_role") or label.get("maker_taker_role") or "UNKNOWN"),
            "join_status": str(join.get("join_status") or "UNKNOWN"),
            "candidate_join_status": str(join.get("candidate_join_status") or "UNKNOWN"),
            "order_join_status": str(join.get("order_join_status") or "UNKNOWN"),
            "future_reference_price_source": str(label.get("future_reference_price_source") or "UNKNOWN"),
            "observed_run_path": str(observed_run),
            "join_run_path": str(join.get("run_id") or ""),
        }
        markouts.append(markout)
        fill_key = f"{source_sha}::{fill_id_s}"
        manifest_record = split_manifest.get(fill_key)
        if manifest_record is None:
            split_manifest[fill_key] = {
                "fill_split_key": fill_key,
                "source_telemetry_sha256": source_sha,
                "fill_id": fill_id_s,
                "holdout_split": split,
                "venue_id": venue_id,
                "side": side,
                "maker_taker_role": markout["maker_taker_role"],
                "join_status": markout["join_status"],
                "candidate_join_status": markout["candidate_join_status"],
                "order_join_status": markout["order_join_status"],
                "future_reference_price_source_list": [markout["future_reference_price_source"]],
                "markout_horizon_ms_list": [horizon_ms],
                "markout_row_count": 1,
            }
        else:
            if manifest_record["holdout_split"] != split:
                raise ValueError(f"conflicting holdout_split for fill_id {fill_id_s}")
            if horizon_ms not in manifest_record["markout_horizon_ms_list"]:
                manifest_record["markout_horizon_ms_list"].append(horizon_ms)
                manifest_record["markout_horizon_ms_list"].sort()
            if markout["future_reference_price_source"] not in manifest_record["future_reference_price_source_list"]:
                manifest_record["future_reference_price_source_list"].append(markout["future_reference_price_source"])
                manifest_record["future_reference_price_source_list"].sort()
            manifest_record["markout_row_count"] += 1
    return summary, markouts, split_manifest


def _bucket_dimensions(record: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    horizon = str(record["markout_horizon_ms"])
    venue = str(record["venue_id"])
    side = str(record["side"])
    split = str(record["holdout_split"])
    return [
        ("GLOBAL", {"scope": "GLOBAL"}),
        (f"HORIZON:{horizon}", {"markout_horizon_ms": horizon}),
        (f"HORIZON_VENUE:{horizon}:{venue}", {"markout_horizon_ms": horizon, "venue_id": venue}),
        (
            f"HORIZON_VENUE_SIDE:{horizon}:{venue}:{side}",
            {"markout_horizon_ms": horizon, "venue_id": venue, "side": side},
        ),
        (
            f"HORIZON_VENUE_SIDE_SPLIT:{horizon}:{venue}:{side}:{split}",
            {
                "markout_horizon_ms": horizon,
                "venue_id": venue,
                "side": side,
                "holdout_split": split,
            },
        ),
    ]


def _bucket_record(
    *,
    run_id: str,
    seq: int,
    timestamp_ns: int,
    bucket_id: str,
    bucket_dimensions: dict[str, str],
    records: list[dict[str, Any]],
    min_fills_per_bucket: int,
    min_holdout_fills_per_bucket: int,
) -> dict[str, Any]:
    values = sorted(float(record["markout_pnl"]) for record in records)
    train_values = [float(record["markout_pnl"]) for record in records if record["holdout_split"] == "TRAIN"]
    holdout_values = [float(record["markout_pnl"]) for record in records if record["holdout_split"] == "HOLDOUT"]
    fill_ids = {record["fill_id"] for record in records}
    train_fill_ids = {record["fill_id"] for record in records if record["holdout_split"] == "TRAIN"}
    holdout_fill_ids = {record["fill_id"] for record in records if record["holdout_split"] == "HOLDOUT"}
    adverse_count = sum(1 for value in values if value < 0.0)
    maker_taker_counts = _count_map(records, "maker_taker_role")
    candidate_join_counts = _count_map(records, "candidate_join_status")
    future_reference_counts = _count_map(records, "future_reference_price_source")
    sparse = len(fill_ids) < min_fills_per_bucket or len(holdout_fill_ids) < min_holdout_fills_per_bucket
    reasons: list[str] = []
    if sparse:
        reasons.append("sparse_markout_calibration_bucket")
    if maker_taker_counts.get("UNKNOWN", 0) > 0:
        reasons.append("maker_taker_unknown_present")
    if candidate_join_counts.get("MISSING", 0) > 0:
        reasons.append("candidate_join_missing_present")
    if set(future_reference_counts) == {"fair_value"}:
        reasons.append("future_reference_source_fair_value_only")
    reasons.append("requires_feature_rich_markout_model_and_board_review")
    train_mean = _mean(train_values)
    holdout_mean = _mean(holdout_values)
    return {
        "schema_version": 1,
        "label_type": "MARKOUT_CALIBRATION_BUCKET_READINESS",
        "bucket_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "bucket_id": bucket_id,
        "bucket_dimensions": bucket_dimensions,
        "gate_status": "HOLD",
        "gate_reasons": reasons,
        "min_fills_per_bucket": min_fills_per_bucket,
        "min_holdout_fills_per_bucket": min_holdout_fills_per_bucket,
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
        "markout_row_count": len(records),
        "unique_fill_count": len(fill_ids),
        "train_markout_row_count": len(train_values),
        "train_fill_count": len(train_fill_ids),
        "holdout_markout_row_count": len(holdout_values),
        "holdout_fill_count": len(holdout_fill_ids),
        "adverse_count": adverse_count,
        "adverse_rate": adverse_count / len(values) if values else None,
        "mean_markout_pnl": _mean(values),
        "train_mean_markout_pnl": train_mean,
        "holdout_mean_markout_pnl": holdout_mean,
        "train_holdout_mean_delta": (
            holdout_mean - train_mean
            if holdout_mean is not None and train_mean is not None
            else None
        ),
        "min_markout_pnl": values[0] if values else None,
        "p05_markout_pnl": _quantile(values, 0.05),
        "p50_markout_pnl": _quantile(values, 0.50),
        "p95_markout_pnl": _quantile(values, 0.95),
        "max_markout_pnl": values[-1] if values else None,
        "maker_taker_role_counts": maker_taker_counts,
        "join_status_counts": _count_map(records, "join_status"),
        "candidate_join_status_counts": candidate_join_counts,
        "order_join_status_counts": _count_map(records, "order_join_status"),
        "future_reference_price_source_counts": future_reference_counts,
        "source_telemetry_sha256_counts": _count_map(records, "source_telemetry_sha256"),
    }


def _gate_reason(
    *,
    markout_rows: list[dict[str, Any]],
    split_manifest: dict[str, dict[str, Any]],
    bucket_records: list[dict[str, Any]],
) -> str:
    if not split_manifest:
        return "markout_readiness_missing_fills"
    if not markout_rows:
        return "markout_readiness_missing_markouts"
    if not any(record["holdout_split"] == "HOLDOUT" for record in split_manifest.values()):
        return "markout_readiness_missing_holdout_fills"
    if any("sparse_markout_calibration_bucket" in record["gate_reasons"] for record in bucket_records):
        return "markout_readiness_sparse_buckets"
    if any("maker_taker_unknown_present" in record["gate_reasons"] for record in bucket_records):
        return "markout_readiness_partial_maker_taker_attribution"
    if any("candidate_join_missing_present" in record["gate_reasons"] for record in bucket_records):
        return "markout_readiness_partial_candidate_join"
    return "markout_readiness_requires_feature_rich_model_and_board_review"


def build_markout_calibration_readiness(
    *,
    observed_runs: list[Path],
    join_holdout_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    min_fills_per_bucket: int,
    min_holdout_fills_per_bucket: int,
) -> Path:
    if not observed_runs:
        raise ValueError("at least one --observed-run is required")
    if len(observed_runs) != len(join_holdout_runs):
        raise ValueError("--observed-run and --join-holdout-run must be supplied in matching counts/order")
    run_id = run_id or f"PHASE51C-MARKOUT-CALIBRATION-READINESS-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    input_summaries: list[dict[str, Any]] = []
    all_markouts: list[dict[str, Any]] = []
    split_manifest: dict[str, dict[str, Any]] = {}
    source_sha_list: set[str] = set()
    for observed_run, join_run in zip(observed_runs, join_holdout_runs):
        join_summary, joined_by_fill = _load_join_run(join_run)
        source_sha = str(join_summary["source_telemetry_sha256"])
        observed_summary, markouts, manifest_records = _load_observed_markouts(
            observed_run,
            joined_by_fill,
            expected_source_sha=source_sha,
        )
        source_sha_list.add(source_sha)
        all_markouts.extend(markouts)
        for key, manifest_record in manifest_records.items():
            if key in split_manifest:
                raise ValueError(f"duplicate fill split key across input runs: {key}")
            split_manifest[key] = manifest_record
        input_summaries.append({
            "observed_run": str(observed_run),
            "observed_run_id": observed_summary.get("run_id"),
            "observed_gate_status": observed_summary.get("gate_status"),
            "observed_gate_reason": observed_summary.get("gate_reason"),
            "observed_label_summary_sha256": _sha256_file(observed_run / "observed_label_summary.json"),
            "observed_labels_sha256": _sha256_file(observed_run / "labels.jsonl"),
            "join_holdout_run": str(join_run),
            "join_run_id": join_summary.get("run_id"),
            "join_gate_status": join_summary.get("gate_status"),
            "join_gate_reason": join_summary.get("gate_reason"),
            "join_holdout_summary_sha256": _sha256_file(join_run / "join_holdout_summary.json"),
            "joined_labels_sha256": _sha256_file(join_run / "joined_labels.jsonl"),
            "source_telemetry_sha256": source_sha,
            "markout_rows_selected": len(markouts),
            "fill_rows_selected": len(manifest_records),
        })

    bucket_payloads: dict[str, dict[str, Any]] = {}
    for markout in all_markouts:
        for bucket_id, dimensions in _bucket_dimensions(markout):
            bucket_payload = bucket_payloads.setdefault(bucket_id, {"dimensions": dimensions, "records": []})
            bucket_payload["records"].append(markout)
    bucket_records = [
        _bucket_record(
            run_id=run_id,
            seq=seq,
            timestamp_ns=timestamp_ns,
            bucket_id=bucket_id,
            bucket_dimensions=payload["dimensions"],
            records=payload["records"],
            min_fills_per_bucket=min_fills_per_bucket,
            min_holdout_fills_per_bucket=min_holdout_fills_per_bucket,
        )
        for seq, (bucket_id, payload) in enumerate(sorted(bucket_payloads.items()), start=1)
    ]

    split_manifest_records = [split_manifest[key] for key in sorted(split_manifest)]
    values = [float(record["markout_pnl"]) for record in all_markouts]
    gate_reason = _gate_reason(
        markout_rows=all_markouts,
        split_manifest=split_manifest,
        bucket_records=bucket_records,
    )
    buckets_path = out_dir / "markout_calibration_buckets.jsonl"
    split_manifest_path = out_dir / "markout_fill_split_manifest.jsonl"
    summary_path = out_dir / "markout_calibration_readiness_summary.json"
    _write_jsonl(buckets_path, bucket_records)
    _write_jsonl(split_manifest_path, split_manifest_records)
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
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "source_telemetry_sha256_list": sorted(source_sha_list),
        "input_run_pairs": input_summaries,
        "min_fills_per_bucket": min_fills_per_bucket,
        "min_holdout_fills_per_bucket": min_holdout_fills_per_bucket,
        "bucket_count": len(bucket_records),
        "markout_row_count": len(all_markouts),
        "unique_fill_count": len(split_manifest),
        "train_fill_count": sum(1 for record in split_manifest.values() if record["holdout_split"] == "TRAIN"),
        "holdout_fill_count": sum(1 for record in split_manifest.values() if record["holdout_split"] == "HOLDOUT"),
        "markout_horizon_ms_list": sorted({record["markout_horizon_ms"] for record in all_markouts}),
        "adverse_count": sum(1 for value in values if value < 0.0),
        "adverse_rate": (
            sum(1 for value in values if value < 0.0) / len(values)
            if values
            else None
        ),
        "mean_markout_pnl": _mean(values),
        "maker_taker_role_counts_by_fill": _fill_count_map(split_manifest, "maker_taker_role"),
        "candidate_join_status_counts_by_fill": _fill_count_map(split_manifest, "candidate_join_status"),
        "future_reference_price_source_counts_by_fill": _fill_list_value_count_map(
            split_manifest,
            "future_reference_price_source_list",
        ),
        "markout_fill_split_manifest_count": len(split_manifest_records),
        "markout_fill_split_manifest_sha256": _sha256_file(split_manifest_path),
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
    parser.add_argument("--observed-run", type=Path, action="append", required=True)
    parser.add_argument("--join-holdout-run", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--min-fills-per-bucket", type=int, default=200)
    parser.add_argument("--min-holdout-fills-per-bucket", type=int, default=50)
    args = parser.parse_args()
    try:
        out_dir = build_markout_calibration_readiness(
            observed_runs=args.observed_run,
            join_holdout_runs=args.join_holdout_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            min_fills_per_bucket=args.min_fills_per_bucket,
            min_holdout_fills_per_bucket=args.min_holdout_fills_per_bucket,
        )
    except Exception as exc:
        print(f"phase51c_markout_calibration_readiness: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_markout_calibration_readiness: wrote {out_dir}")
    print("phase51c_markout_calibration_readiness: status HOLD (markout readiness only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
