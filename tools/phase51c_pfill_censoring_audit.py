#!/usr/bin/env python3
"""Audit Phase 5.1c P_fill censoring without approving model training.

This offline evidence gate consumes one or more Phase 5.1c P_fill outcome
packs and classifies censored order labels into deterministic diagnostic
reasons. It does not train a model, submit orders, approve EV admission, or
make financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_pfill_censoring_audit"
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
OBSERVED_REASONS = {
    "OBSERVED_FILLED",
    "OBSERVED_NOT_FILLED_TO_TERMINAL",
}
CENSOR_REASONS = {
    "SOURCE_WINDOW_BOUNDARY",
    "NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW",
    "ORDER_IDENTITY_GAP",
    "FILL_JOIN_GAP",
    "VENUE_SCOPE_GAP",
    "PARSER_GAP",
    "UNKNOWN_REQUIRES_REVIEW",
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


def _resolve_path(path_text: Any, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else base_dir / str(path_text)


def _safe_int(value: Any) -> int | None:
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


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _load_label_lake_record_count(run_path: Path, summary: dict[str, Any]) -> int | None:
    label_lake_run = _resolve_path(summary.get("label_lake_run"), run_path)
    if label_lake_run is None:
        return None
    label_lake_summary_path = label_lake_run / "label_lake_summary.json"
    if not label_lake_summary_path.exists():
        return None
    label_lake_summary = _load_json(label_lake_summary_path)
    if label_lake_summary.get("source_telemetry_sha256") != summary.get("source_telemetry_sha256"):
        raise ValueError(f"{label_lake_summary_path} source_telemetry_sha256 mismatch")
    if label_lake_summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{label_lake_summary_path} baseline_commit mismatch")
    _check_unsafe(label_lake_summary, label_lake_summary_path, label="summary")
    return _safe_int(label_lake_summary.get("record_count"))


def _load_outcome_run(run_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int | None]:
    summary_path = run_path / "pfill_outcome_summary.json"
    labels_path = run_path / "pfill_order_labels.jsonl"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    if not summary.get("source_telemetry_sha256"):
        raise ValueError(f"{summary_path} missing source_telemetry_sha256")
    _check_unsafe(summary, summary_path, label="summary")
    labels: list[dict[str, Any]] = []
    for _, record in _iter_jsonl(labels_path):
        if record.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(record, labels_path, label="label")
        labels.append(record)
    return summary, labels, _load_label_lake_record_count(run_path, summary)


def _classify_censored(
    label: dict[str, Any],
    *,
    source_record_count: int | None,
    boundary_source_line_margin: int,
) -> tuple[str, str]:
    if not label.get("order_key") or label.get("order_label_seq") is None:
        return "PARSER_GAP", "missing order_key or order_label_seq"
    order_source_line = _safe_int(label.get("order_source_line"))
    order_source_t = _safe_int(label.get("order_source_t"))
    if order_source_line is None or order_source_t is None:
        return "PARSER_GAP", "missing order source position"
    if not label.get("order_id_hash") and not label.get("client_order_id_hash"):
        return "ORDER_IDENTITY_GAP", "missing order_id_hash and client_order_id_hash"
    if _safe_int(label.get("fill_count")) not in (None, 0):
        return "FILL_JOIN_GAP", "censored label carries non-zero fill_count"
    if _safe_int(label.get("terminal_event_count")) not in (None, 0):
        return "PARSER_GAP", "censored label carries terminal events"
    if source_record_count is not None:
        remaining_source_lines = source_record_count - order_source_line
        if remaining_source_lines < 0:
            return "PARSER_GAP", "order source line beyond source record count"
        if remaining_source_lines <= boundary_source_line_margin:
            return "SOURCE_WINDOW_BOUNDARY", "order is too near source window end"
        return "NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW", "no fill or terminal event despite available source window"
    return "UNKNOWN_REQUIRES_REVIEW", "source record count unavailable"


def _outcome_reason(
    label: dict[str, Any],
    *,
    source_record_count: int | None,
    boundary_source_line_margin: int,
) -> tuple[str, str, bool]:
    status = str(label.get("outcome_status") or "")
    if status == "OBSERVED_FILLED":
        return "OBSERVED_FILLED", "observed fill label", False
    if status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
        return "OBSERVED_NOT_FILLED_TO_TERMINAL", "observed terminal not-fill label", False
    if status != "CENSORED_OR_UNOBSERVED":
        raise ValueError(f"unexpected outcome_status={status!r}")
    reason, detail = _classify_censored(
        label,
        source_record_count=source_record_count,
        boundary_source_line_margin=boundary_source_line_margin,
    )
    return reason, detail, True


def _bucket_key(record: dict[str, Any]) -> tuple[str, dict[str, str]]:
    dimensions = {
        "source_telemetry_sha256": str(record.get("source_telemetry_sha256") or "UNKNOWN"),
        "venue_id": str(record.get("venue_id") or "UNKNOWN"),
        "side": str(record.get("side") or "UNKNOWN"),
        "order_holdout_split": str(record.get("order_holdout_split") or "UNKNOWN"),
        "outcome_status": str(record.get("outcome_status") or "UNKNOWN"),
        "censor_reason": str(record.get("censor_reason") or "UNKNOWN"),
    }
    return _stable_hash(dimensions), dimensions


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def build_pfill_censoring_audit(
    *,
    pfill_outcome_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    boundary_source_line_margin: int,
) -> Path:
    if not pfill_outcome_runs:
        raise ValueError("at least one --pfill-outcome-run is required")
    if boundary_source_line_margin < 0:
        raise ValueError("--boundary-source-line-margin must be >= 0")
    run_id = run_id or f"PHASE51C-PFILL-CENSORING-AUDIT-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    labels_out: list[dict[str, Any]] = []
    bucket_payloads: dict[str, dict[str, Any]] = {}
    input_summaries: list[dict[str, Any]] = []
    total = {
        "order_label_count": 0,
        "observed_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "censored_count": 0,
        "holdout_count": 0,
        "train_count": 0,
    }
    reason_counts: dict[str, int] = {}
    source_sha: set[str] = set()
    split_by_source_order: dict[tuple[str, str], str] = {}
    for run_path in pfill_outcome_runs:
        summary, labels, source_record_count = _load_outcome_run(run_path)
        expected_counts = {
            "order_label_count": int(summary.get("order_label_count") or 0),
            "filled_count": int(summary.get("filled_count") or 0),
            "not_filled_count": int(summary.get("not_filled_count") or 0),
            "censored_count": int(summary.get("censored_count") or 0),
        }
        actual_counts = {
            "order_label_count": len(labels),
            "filled_count": sum(1 for label in labels if label.get("outcome_status") == "OBSERVED_FILLED"),
            "not_filled_count": sum(
                1 for label in labels if label.get("outcome_status") == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL"
            ),
            "censored_count": sum(1 for label in labels if label.get("outcome_status") == "CENSORED_OR_UNOBSERVED"),
        }
        if actual_counts != expected_counts:
            raise ValueError(f"{run_path} summary counts do not reconcile: {actual_counts} != {expected_counts}")
        source_sha.add(str(summary["source_telemetry_sha256"]))
        input_summaries.append({
            "run_path": str(run_path),
            "run_id": summary.get("run_id"),
            "gate_status": summary.get("gate_status"),
            "gate_reason": summary.get("gate_reason"),
            "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
            "source_record_count": source_record_count,
            "pfill_outcome_summary_sha256": _sha256_file(run_path / "pfill_outcome_summary.json"),
            "pfill_order_labels_sha256": _sha256_file(run_path / "pfill_order_labels.jsonl"),
            **expected_counts,
        })
        for label in labels:
            reason, detail, censored = _outcome_reason(
                label,
                source_record_count=source_record_count,
                boundary_source_line_margin=boundary_source_line_margin,
            )
            if reason not in OBSERVED_REASONS and reason not in CENSOR_REASONS:
                raise ValueError(f"unexpected censor reason {reason}")
            order_key = str(label.get("order_key") or "")
            source_key = str(label.get("source_telemetry_sha256") or summary.get("source_telemetry_sha256"))
            split = str(label.get("order_holdout_split") or "UNKNOWN").upper()
            split_key = (source_key, order_key)
            existing_split = split_by_source_order.get(split_key)
            if existing_split is not None and existing_split != split:
                raise ValueError(f"conflicting order_holdout_split for order_key {order_key}")
            split_by_source_order.setdefault(split_key, split)
            record = {
                "schema_version": 1,
                "label_type": "PFILL_CENSORING_AUDIT_LABEL",
                "run_id": run_id,
                "baseline_commit": BASELINE_COMMIT,
                "timestamp_local_ns": timestamp_ns + len(labels_out) + 1,
                "source_run_id": label.get("run_id"),
                "source_run_path": str(run_path),
                "source_telemetry_sha256": source_key,
                "order_key": order_key,
                "order_label_seq": label.get("order_label_seq"),
                "order_source_line": label.get("order_source_line"),
                "order_source_t": label.get("order_source_t"),
                "source_record_count": source_record_count,
                "venue_id": label.get("venue_id"),
                "side": _canonical_side(label.get("side")),
                "order_holdout_split": split,
                "outcome_status": label.get("outcome_status"),
                "p_fill_outcome": label.get("p_fill_outcome"),
                "terminal_event_count": label.get("terminal_event_count"),
                "terminal_action_first": label.get("terminal_action_first"),
                "fill_count": label.get("fill_count"),
                "censored": censored,
                "censor_reason": reason,
                "censor_reason_detail": detail,
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
            }
            labels_out.append(record)
            total["order_label_count"] += 1
            total["holdout_count" if split == "HOLDOUT" else "train_count"] += 1
            if censored:
                total["censored_count"] += 1
            else:
                total["observed_count"] += 1
                if reason == "OBSERVED_FILLED":
                    total["filled_count"] += 1
                else:
                    total["not_filled_count"] += 1
            _increment(reason_counts, reason)
            bucket_id, dimensions = _bucket_key(record)
            bucket = bucket_payloads.setdefault(bucket_id, {"bucket_dimensions": dimensions, "counts": {}})
            _increment(bucket["counts"], "order_label_count")
            _increment(bucket["counts"], "censored_count" if censored else "observed_count")

    bucket_records: list[dict[str, Any]] = []
    for seq, (bucket_id, bucket) in enumerate(sorted(bucket_payloads.items()), start=1):
        counts = bucket["counts"]
        bucket_records.append({
            "schema_version": 1,
            "label_type": "PFILL_CENSORING_AUDIT_BUCKET",
            "bucket_seq": seq,
            "timestamp_local_ns": timestamp_ns + seq,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "bucket_id": bucket_id,
            "bucket_dimensions": bucket["bucket_dimensions"],
            "gate_status": "HOLD",
            "no_live_flag": True,
            "approved_for_model_training": False,
            "approved_for_live": False,
            "admissible_for_financial_claim": False,
            "admissible_for_ev_admission": False,
            **counts,
        })

    if total["order_label_count"] != total["observed_count"] + total["censored_count"]:
        raise ValueError("total order labels do not reconcile to observed+censored")
    gate_reason = (
        "pfill_censoring_audit_unknowns_present"
        if reason_counts.get("UNKNOWN_REQUIRES_REVIEW", 0) > 0
        else "pfill_censoring_audit_censored_orders_classified"
    )
    labels_path = out_dir / "pfill_censoring_labels.jsonl"
    buckets_path = out_dir / "pfill_censoring_buckets.jsonl"
    summary_path = out_dir / "pfill_censoring_audit_summary.json"
    _write_jsonl(labels_path, labels_out)
    _write_jsonl(buckets_path, bucket_records)
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
        "source_telemetry_sha256_list": sorted(source_sha),
        "input_pfill_outcome_runs": input_summaries,
        "boundary_source_line_margin": boundary_source_line_margin,
        "reason_counts": dict(sorted(reason_counts.items())),
        "bucket_count": len(bucket_records),
        "pfill_censoring_labels_sha256": _sha256_file(labels_path),
        "pfill_censoring_buckets_sha256": _sha256_file(buckets_path),
        **total,
        "censored_rate": total["censored_count"] / total["order_label_count"] if total["order_label_count"] else None,
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, buckets_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, buckets_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfill-outcome-run", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--boundary-source-line-margin", type=int, default=1000)
    args = parser.parse_args()
    try:
        out_dir = build_pfill_censoring_audit(
            pfill_outcome_runs=args.pfill_outcome_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            boundary_source_line_margin=args.boundary_source_line_margin,
        )
    except Exception as exc:
        print(f"phase51c_pfill_censoring_audit: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_pfill_censoring_audit: wrote {out_dir}")
    print("phase51c_pfill_censoring_audit: status HOLD (diagnostic censoring audit only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
