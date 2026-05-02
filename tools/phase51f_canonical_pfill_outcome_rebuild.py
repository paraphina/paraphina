#!/usr/bin/env python3
"""Build the Phase 5.1f canonical P_fill outcome review pack.

This is an offline evidence gate. It consumes a Phase 5.1e lifecycle-truth
audit pack plus the source Phase 5.1c P_fill outcome runs, then emits one
P_fill row per canonical lifecycle group. It does not train a model, submit
orders, approve EV admission, approve live/canary use, or make financial
claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51f_canonical_pfill_outcome"

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

OBSERVED_FILLED_STATUSES = {
    "STAYS_FILLED",
    "CENSORED_TO_CANONICAL_FILLED_REVIEW",
}
OBSERVED_NOT_FILLED_STATUSES = {
    "STAYS_NOT_FILLED",
    "CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW",
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


def _redacted_decision_id_status(pfill_labels: list[dict[str, Any]]) -> tuple[bool, str | None]:
    decision_ids = sorted({
        str(label.get("decision_id"))
        for label in pfill_labels
        if label.get("decision_id") not in (None, "")
    })
    return bool(decision_ids), _stable_hash(["decision_id", decision_ids]) if decision_ids else None


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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _holdout_split(stable_key: str) -> str:
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
    return "HOLDOUT" if int(digest[:8], 16) % 5 == 0 else "TRAIN"


def _empty_counts() -> dict[str, int]:
    return {
        "order_label_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "censored_count": 0,
        "train_count": 0,
        "holdout_count": 0,
    }


def _base_label(run_id: str, seq: int, timestamp_ns: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_seq": seq,
        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
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
    }


def _load_pfill_runs(pfill_outcome_runs: list[Path]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if not pfill_outcome_runs:
        raise ValueError("at least one --pfill-outcome-run is required")
    input_summaries: list[dict[str, Any]] = []
    labels_by_source_order: dict[tuple[str, str], dict[str, Any]] = {}
    for run_path in pfill_outcome_runs:
        run_path = _resolve_run_path(run_path)
        summary = _load_hold_summary(run_path, "pfill_outcome_summary.json")
        labels_path = run_path / "pfill_order_labels.jsonl"
        labels: list[dict[str, Any]] = []
        for _, label in _iter_jsonl(labels_path):
            if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
                continue
            _check_unsafe(label, labels_path, label="label")
            labels.append(label)
            source_sha = str(label.get("source_telemetry_sha256") or summary.get("source_telemetry_sha256") or "")
            order_key = str(label.get("order_key") or "")
            if not source_sha or not order_key:
                raise ValueError(f"{labels_path} missing source_telemetry_sha256/order_key")
            key = (source_sha, order_key)
            if key in labels_by_source_order:
                raise ValueError(f"duplicate source order key {key} in P_fill inputs")
            labels_by_source_order[key] = label
        expected = {
            "order_label_count": int(summary.get("order_label_count") or 0),
            "filled_count": int(summary.get("filled_count") or 0),
            "not_filled_count": int(summary.get("not_filled_count") or 0),
            "censored_count": int(summary.get("censored_count") or 0),
        }
        actual = {
            "order_label_count": len(labels),
            "filled_count": sum(1 for row in labels if row.get("outcome_status") == "OBSERVED_FILLED"),
            "not_filled_count": sum(1 for row in labels if row.get("outcome_status") == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL"),
            "censored_count": sum(1 for row in labels if row.get("outcome_status") == "CENSORED_OR_UNOBSERVED"),
        }
        if actual != expected:
            raise ValueError(f"{run_path} summary counts do not reconcile: {actual} != {expected}")
        input_summaries.append({
            "run_path": str(run_path),
            "run_id": summary.get("run_id"),
            "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
            "pfill_outcome_summary_sha256": _sha256_file(run_path / "pfill_outcome_summary.json"),
            "pfill_order_labels_sha256": _sha256_file(labels_path),
            **expected,
        })
    return input_summaries, labels_by_source_order


def _load_lifecycle_truth_run(lifecycle_truth_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lifecycle_truth_run = _resolve_run_path(lifecycle_truth_run)
    summary = _load_hold_summary(lifecycle_truth_run, "lifecycle_truth_audit_summary.json")
    labels_path = lifecycle_truth_run / "order_lifecycle_truth_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        labels.append(label)
    if len(labels) != int(summary.get("order_label_count") or 0):
        raise ValueError(f"{labels_path} row count does not match lifecycle summary")
    return summary, labels


def _representative(rows: list[dict[str, Any]], pfill_labels: list[dict[str, Any]]) -> dict[str, Any]:
    def sort_key(item: tuple[dict[str, Any], dict[str, Any] | None]) -> tuple[int, int, str]:
        row, pfill = item
        status = str(row.get("canonical_status") or "")
        rank = 0 if status in OBSERVED_FILLED_STATUSES else 1 if status in OBSERVED_NOT_FILLED_STATUSES else 2
        source_line = pfill.get("order_source_line") if pfill else row.get("order_source_line")
        try:
            line = int(source_line)
        except (TypeError, ValueError):
            line = 10**18
        return rank, line, str(row.get("order_key") or "")

    pfill_by_order = {str(label.get("order_key")): label for label in pfill_labels}
    row, pfill = sorted(((row, pfill_by_order.get(str(row.get("order_key")))) for row in rows), key=sort_key)[0]
    merged = dict(row)
    if pfill:
        for key in (
            "price",
            "size",
            "order_id_hash",
            "client_order_id_hash",
            "fill_count",
            "first_fill_time_ms",
            "last_fill_time_ms",
            "filled_size_total",
            "maker_taker_role_counts",
            "terminal_action_first",
            "terminal_source_line_first",
            "terminal_source_t_first",
            "observed_horizon_source_ticks",
        ):
            if key in pfill:
                merged[key] = pfill[key]
    return merged


def _canonical_outcome(rows: list[dict[str, Any]]) -> tuple[float | None, str, str, str]:
    statuses = {str(row.get("canonical_status") or "") for row in rows}
    if statuses & OBSERVED_FILLED_STATUSES:
        return 1.0, "OBSERVED_FILLED", "CANONICAL_OBSERVED_FILLED", "canonical lifecycle group has fill evidence"
    if statuses & OBSERVED_NOT_FILLED_STATUSES:
        return (
            0.0,
            "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL",
            "CANONICAL_OBSERVED_NOT_FILLED",
            "canonical lifecycle group has direct terminal evidence",
        )
    return None, "CENSORED_OR_UNOBSERVED", "CANONICAL_REVIEW_QUARANTINED", "canonical lifecycle group requires manual review"


def _group_source_status(rows: list[dict[str, Any]], pfill_labels: list[dict[str, Any]]) -> dict[str, Any]:
    old_splits = sorted({
        str(label.get("order_holdout_split") or "UNKNOWN").upper()
        for label in pfill_labels
        if label.get("order_holdout_split") is not None
    })
    return {
        "source_label_count": len(rows),
        "source_order_keys": [str(row.get("order_key")) for row in sorted(rows, key=lambda r: str(r.get("order_key") or ""))],
        "source_current_status_counts": _counts(str(row.get("current_outcome_status") or "UNKNOWN") for row in rows),
        "source_canonical_status_counts": _counts(str(row.get("canonical_status") or "UNKNOWN") for row in rows),
        "source_old_split_values": old_splits,
        "source_old_split_conflict": len(old_splits) > 1,
    }


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _terminal_action_for_label(label: dict[str, Any], rows: list[dict[str, Any]], outcome_status: str) -> str | None:
    if outcome_status == "OBSERVED_FILLED":
        return None
    if outcome_status != "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
        return label.get("terminal_action_first")
    if label.get("terminal_action_first"):
        return str(label.get("terminal_action_first"))
    if any(str(row.get("canonical_status") or "") == "CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW" for row in rows):
        return "canonical_direct_terminal"
    return "cancel"


def build_canonical_pfill_outcomes(
    *,
    lifecycle_truth_run: Path,
    pfill_outcome_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51F-CANONICAL-PFILL-OUTCOME-REBUILD-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    lifecycle_summary, truth_rows = _load_lifecycle_truth_run(lifecycle_truth_run)
    pfill_input_summaries, pfill_by_source_order = _load_pfill_runs(pfill_outcome_runs)

    source_rows_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    pfill_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in truth_rows:
        source_sha = str(row.get("source_telemetry_sha256") or "")
        order_key = str(row.get("order_key") or "")
        group_id = str(row.get("canonical_group_id") or "")
        if not source_sha or not order_key or not group_id:
            raise ValueError("lifecycle truth row missing source_telemetry_sha256/order_key/canonical_group_id")
        pfill_label = pfill_by_source_order.get((source_sha, order_key))
        if pfill_label is None:
            raise ValueError(f"lifecycle truth row missing source P_fill label for {(source_sha, order_key)}")
        group_key = (source_sha, group_id)
        source_rows_by_group.setdefault(group_key, []).append(row)
        pfill_by_group.setdefault(group_key, []).append(pfill_label)

    lifecycle_graph_group_count = int(lifecycle_summary.get("canonical_group_count") or 0)

    labels: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    split_conflicts: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    counts = _empty_counts()
    canonical_status_counts: dict[str, int] = {}
    canonical_review_status_counts: dict[str, int] = {}

    for seq, group_key in enumerate(sorted(source_rows_by_group), start=1):
        source_sha, group_id = group_key
        rows = source_rows_by_group[group_key]
        pfill_labels = pfill_by_group[group_key]
        rep = _representative(rows, pfill_labels)
        outcome, outcome_status, canonical_review_status, reason = _canonical_outcome(rows)
        canonical_order_key = _stable_hash(["phase51f", source_sha, group_id])
        split = _holdout_split(canonical_order_key)
        group_source = _group_source_status(rows, pfill_labels)
        decision_id_present, decision_id_hash = _redacted_decision_id_status(pfill_labels)
        if group_source["source_old_split_conflict"]:
            split_conflicts.append({
                "canonical_order_key": canonical_order_key,
                "source_telemetry_sha256": source_sha,
                "canonical_group_id": group_id,
                **group_source,
            })
        terminal_action = _terminal_action_for_label(rep, rows, outcome_status)
        label = _base_label(run_id, seq, timestamp_ns)
        label.update({
            "source": "phase51e_lifecycle_truth_audit_and_phase51c_pfill_outcome",
            "source_telemetry_sha256": source_sha,
            "lifecycle_truth_run": str(_resolve_run_path(lifecycle_truth_run)),
            "canonical_group_id": group_id,
            "order_key": canonical_order_key,
            "order_holdout_split": split,
            "source_pfill_run_ids": sorted({str(row.get("source_pfill_run_id") or "") for row in rows}),
            "source_pfill_run_paths": sorted({str(row.get("source_pfill_run_path") or "") for row in rows}),
            "canonical_review_status": canonical_review_status,
            "canonical_review_reason": reason,
            "canonical_training_admissibility": "REVIEW_GATE_ONLY_NOT_APPROVED_FOR_MODEL_TRAINING",
            "p_fill_outcome": outcome,
            "outcome_status": outcome_status,
            "label_status": (
                "CANONICAL_ORDER_PFILL_OUTCOME"
                if outcome is not None
                else "CANONICAL_ORDER_PFILL_REVIEW_QUARANTINE"
            ),
            "training_hold_reason": "requires_phase51f_board_review_before_model_training",
            "order_label_seq": rep.get("order_label_seq"),
            "order_source_line": rep.get("order_source_line"),
            "order_source_t": rep.get("order_source_t"),
            "venue_id": rep.get("venue_id"),
            "side": rep.get("side"),
            "price": rep.get("price"),
            "size": rep.get("size"),
            "decision_id_present": decision_id_present,
            "decision_id_hash": decision_id_hash,
            "order_id_hash": rep.get("order_id_hash"),
            "client_order_id_hash": rep.get("client_order_id_hash"),
            "fill_count": max((int(label.get("fill_count") or 0) for label in pfill_labels), default=0),
            "first_fill_time_ms": rep.get("first_fill_time_ms"),
            "last_fill_time_ms": rep.get("last_fill_time_ms"),
            "filled_size_total": rep.get("filled_size_total"),
            "maker_taker_role_counts": rep.get("maker_taker_role_counts"),
            "terminal_event_count": max((int(row.get("canonical_direct_terminal_count") or 0) for row in rows), default=0),
            "terminal_action_first": terminal_action,
            "terminal_source_line_first": rep.get("terminal_source_line_first"),
            "terminal_source_t_first": rep.get("terminal_source_t_first"),
            "observed_horizon_source_ticks": rep.get("observed_horizon_source_ticks"),
            **group_source,
        })
        labels.append(label)
        for row in sorted(rows, key=lambda item: str(item.get("order_key") or "")):
            source_manifest.append({
                "canonical_order_key": canonical_order_key,
                "source_telemetry_sha256": source_sha,
                "canonical_group_id": group_id,
                "source_order_key": row.get("order_key"),
                "source_order_label_seq": row.get("order_label_seq"),
                "source_pfill_run_id": row.get("source_pfill_run_id"),
                "current_outcome_status": row.get("current_outcome_status"),
                "canonical_status": row.get("canonical_status"),
                "canonical_review_status": canonical_review_status,
                "output_outcome_status": outcome_status,
            })
        if outcome is None:
            quarantined.append({
                "canonical_order_key": canonical_order_key,
                "source_telemetry_sha256": source_sha,
                "canonical_group_id": group_id,
                "quarantine_reason": canonical_review_status,
                **group_source,
            })
        counts["order_label_count"] += 1
        counts["holdout_count" if split == "HOLDOUT" else "train_count"] += 1
        if outcome_status == "OBSERVED_FILLED":
            counts["filled_count"] += 1
        elif outcome_status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
            counts["not_filled_count"] += 1
        else:
            counts["censored_count"] += 1
        for status, count in group_source["source_canonical_status_counts"].items():
            canonical_status_counts[status] = canonical_status_counts.get(status, 0) + count
        canonical_review_status_counts[canonical_review_status] = canonical_review_status_counts.get(canonical_review_status, 0) + 1

    if sum(row["source_label_count"] for row in labels) != len(truth_rows):
        raise ValueError("canonical label source count does not reconcile to lifecycle truth rows")

    canonical_labels_path = out_dir / "canonical_pfill_order_labels.jsonl"
    compatibility_labels_path = out_dir / "pfill_order_labels.jsonl"
    source_manifest_path = out_dir / "source_to_canonical_order_manifest.jsonl"
    split_conflicts_path = out_dir / "split_conflict_manifest.jsonl"
    quarantined_path = out_dir / "quarantined_review_labels.jsonl"
    summary_path = out_dir / "pfill_outcome_summary.json"
    canonical_summary_path = out_dir / "canonical_pfill_outcome_summary.json"

    _write_jsonl(canonical_labels_path, labels)
    _write_jsonl(compatibility_labels_path, labels)
    _write_jsonl(source_manifest_path, source_manifest)
    _write_jsonl(split_conflicts_path, split_conflicts)
    _write_jsonl(quarantined_path, quarantined)

    gate_reason = (
        "phase51f_canonical_pfill_contains_quarantined_review_groups"
        if counts["censored_count"] > 0
        else "phase51f_canonical_pfill_requires_board_review"
    )
    source_telemetry_sha256_list = sorted({str(row.get("source_telemetry_sha256") or "") for row in truth_rows})
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
        "lifecycle_truth_run": str(_resolve_run_path(lifecycle_truth_run)),
        "lifecycle_truth_summary_sha256": _sha256_file(_resolve_run_path(lifecycle_truth_run) / "lifecycle_truth_audit_summary.json"),
        "lifecycle_truth_labels_sha256": _sha256_file(_resolve_run_path(lifecycle_truth_run) / "order_lifecycle_truth_labels.jsonl"),
        "input_pfill_outcome_runs": pfill_input_summaries,
        "source_telemetry_sha256": _stable_hash(source_telemetry_sha256_list),
        "source_telemetry_sha256_list": source_telemetry_sha256_list,
        "source_label_count": len(truth_rows),
        "canonical_group_count": len(labels),
        "lifecycle_graph_canonical_group_count": lifecycle_graph_group_count,
        "pfill_group_count_diff_vs_lifecycle_graph": (
            len(labels) - lifecycle_graph_group_count if lifecycle_graph_group_count else None
        ),
        "source_to_canonical_manifest_count": len(source_manifest),
        "quarantined_review_label_count": len(quarantined),
        "split_conflict_count": len(split_conflicts),
        "canonical_status_counts": dict(sorted(canonical_status_counts.items())),
        "canonical_review_status_counts": dict(sorted(canonical_review_status_counts.items())),
        "p_fill_positive_rate_observed": (
            counts["filled_count"] / (counts["filled_count"] + counts["not_filled_count"])
            if counts["filled_count"] + counts["not_filled_count"] > 0
            else None
        ),
        **counts,
    }
    _write_json(summary_path, summary)
    _write_json(canonical_summary_path, summary)

    artifact_paths = [
        canonical_labels_path,
        compatibility_labels_path,
        source_manifest_path,
        split_conflicts_path,
        quarantined_path,
        summary_path,
        canonical_summary_path,
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
    parser.add_argument("--lifecycle-truth-run", type=Path, required=True)
    parser.add_argument("--pfill-outcome-run", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_canonical_pfill_outcomes(
            lifecycle_truth_run=args.lifecycle_truth_run,
            pfill_outcome_runs=args.pfill_outcome_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51f_canonical_pfill_outcome_rebuild: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51f_canonical_pfill_outcome_rebuild: wrote {out_dir}")
    print("phase51f_canonical_pfill_outcome_rebuild: status HOLD (canonical P_fill review only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
