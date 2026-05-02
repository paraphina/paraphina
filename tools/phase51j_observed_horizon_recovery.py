#!/usr/bin/env python3
"""Recover Phase 5.1 observed P_fill source-tick horizons from lifecycle truth.

This is an offline HOLD-only evidence gate. It consumes a redacted Phase 5.1h
feature audit pack plus Phase 5.1e lifecycle truth, then reconstructs terminal
source-tick horizons for canonical not-filled groups. It does not train a
model, submit orders, approve EV admission, approve live/canary use, or make
financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51j_observed_horizon_recovery"

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

PLACE_ACTION = "place"
DIRECT_TERMINAL_ACTIONS = {"cancel", "expire", "expired", "reject", "rejected"}
CHAIN_TERMINAL_ACTIONS = {"replace"}
GLOBAL_TERMINAL_ACTIONS = {"cancel_all"}
TERMINAL_ACTIONS = DIRECT_TERMINAL_ACTIONS | CHAIN_TERMINAL_ACTIONS | GLOBAL_TERMINAL_ACTIONS


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        self.parent.setdefault(node, node)

    def find(self, node: str) -> str:
        self.add(node)
        parent = self.parent[node]
        if parent != node:
            parent = self.find(parent)
            self.parent[node] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


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
            raise ValueError(f"{labels_path} contains unredacted raw identifier input")
        labels.append(label)
    expected = int(summary.get("label_count") or 0)
    if len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary label_count {expected}")
    return summary, labels


def _load_canonical_summary(canonical_pfill_run: Path) -> dict[str, Any]:
    summary_path = canonical_pfill_run / "canonical_pfill_outcome_summary.json"
    if not summary_path.exists():
        summary_path = canonical_pfill_run / "pfill_outcome_summary.json"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    return summary


def _source_order_keys_by_group(canonical_pfill_run: Path) -> dict[str, set[str]]:
    manifest_path = canonical_pfill_run / "source_to_canonical_order_manifest.jsonl"
    index: dict[str, set[str]] = {}
    for _, row in _iter_jsonl(manifest_path):
        canonical_group_id = str(row.get("canonical_group_id") or "")
        source_order_key = str(row.get("source_order_key") or "")
        if not canonical_group_id or not source_order_key:
            raise ValueError(f"{manifest_path} row missing canonical_group_id/source_order_key")
        index.setdefault(canonical_group_id, set()).add(source_order_key)
    return index


def _node_id(source_sha: str, label: dict[str, Any]) -> str:
    return f"{source_sha}:order_label_seq:{label.get('label_seq')}"


def _alias_keys(source_sha: str, label: dict[str, Any]) -> list[tuple[str, str]]:
    venue = str(label.get("venue_id") or "UNKNOWN").lower()
    keys: list[tuple[str, str]] = []
    for field in ("order_id_hash", "client_order_id_hash", "decision_id"):
        value = label.get(field)
        if value:
            keys.append((f"{source_sha}:{venue}:{field}", str(value)))
    return keys


def _action(label: dict[str, Any]) -> str:
    return str(label.get("action") or "UNKNOWN").strip().lower()


def _load_lifecycle_graph(label_lake_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[tuple[str, str], str]]:
    summary = _load_hold_summary(label_lake_run, "label_lake_summary.json")
    source_sha = str(summary.get("source_telemetry_sha256") or "")
    if not source_sha:
        raise ValueError(f"{label_lake_run} missing source_telemetry_sha256")
    uf = UnionFind()
    alias_to_node: dict[tuple[str, str], str] = {}
    lifecycle_by_node: dict[str, dict[str, Any]] = {}
    labels_path = label_lake_run / "labels.jsonl"
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_LIFECYCLE_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        node = _node_id(source_sha, label)
        uf.add(node)
        lifecycle_by_node[node] = label
        for alias in _alias_keys(source_sha, label):
            previous = alias_to_node.get(alias)
            if previous is None:
                alias_to_node[alias] = node
            else:
                uf.union(previous, node)
    group_by_node = {node: uf.find(node) for node in lifecycle_by_node}
    group_index = {alias: uf.find(node) for alias, node in alias_to_node.items()}
    group_index.update({("node", node): group for node, group in group_by_node.items()})
    return summary, lifecycle_by_node, group_index


def _group_timing_facts(
    lifecycle_by_node: dict[str, dict[str, Any]],
    group_index: dict[tuple[str, str], str],
) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for node, label in lifecycle_by_node.items():
        group = group_index[("node", node)]
        fact = facts.setdefault(
            group,
            {
                "canonical_group_id": group,
                "venue_id": label.get("venue_id"),
                "lifecycle_event_count": 0,
                "actions": {},
                "place_source_t_values": [],
                "terminal_source_t_values": [],
                "terminal_action_counts": {},
            },
        )
        action = _action(label)
        fact["lifecycle_event_count"] += 1
        fact["actions"][action] = fact["actions"].get(action, 0) + 1
        source_t = _safe_int(label.get("source_t"))
        if action == PLACE_ACTION and source_t is not None:
            fact["place_source_t_values"].append(source_t)
        if action in TERMINAL_ACTIONS:
            fact["terminal_action_counts"][action] = fact["terminal_action_counts"].get(action, 0) + 1
            if source_t is not None:
                fact["terminal_source_t_values"].append(source_t)
    for fact in facts.values():
        place_values = sorted(set(fact.pop("place_source_t_values")))
        terminal_values = sorted(set(fact.pop("terminal_source_t_values")))
        place_min = min(place_values) if place_values else None
        terminal_after_place = [
            value for value in terminal_values if place_min is None or value >= place_min
        ]
        terminal_min = min(terminal_after_place) if terminal_after_place else None
        fact["place_source_t_min"] = place_min
        fact["terminal_source_t_min"] = terminal_min
        fact["terminal_event_count"] = sum(int(v) for v in fact["terminal_action_counts"].values())
        fact["terminal_horizon_source_ticks"] = (
            terminal_min - place_min
            if place_min is not None and terminal_min is not None and terminal_min >= place_min
            else None
        )
    return facts


def _load_lifecycle_timing_facts(lifecycle_truth_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summary = _load_hold_summary(lifecycle_truth_run, "lifecycle_truth_audit_summary.json")
    source_inputs = summary.get("source_inputs") or []
    if not isinstance(source_inputs, list) or not source_inputs:
        raise ValueError(f"{lifecycle_truth_run} missing source_inputs")
    all_facts: dict[str, dict[str, Any]] = {}
    source_summaries: list[dict[str, Any]] = []
    for source_input in source_inputs:
        label_lake_run = _resolve_path(Path(str(source_input.get("label_lake_run") or "")))
        lake_summary, lifecycle_by_node, group_index = _load_lifecycle_graph(label_lake_run)
        source_sha = str(lake_summary.get("source_telemetry_sha256") or "")
        expected_hash = str(source_input.get("label_lake_labels_sha256") or "")
        labels_path = label_lake_run / "labels.jsonl"
        labels_hash = _sha256_file(labels_path)
        if expected_hash and expected_hash != labels_hash:
            raise ValueError(f"{labels_path} sha256 mismatch against lifecycle truth source input")
        for group_id, fact in _group_timing_facts(lifecycle_by_node, group_index).items():
            if group_id in all_facts:
                raise ValueError(f"duplicate canonical lifecycle group_id={group_id}")
            all_facts[group_id] = fact | {"source_telemetry_sha256": source_sha}
        source_summaries.append({
            "source_telemetry_sha256": source_sha,
            "label_lake_run": str(label_lake_run),
            "label_lake_summary_sha256": _sha256_file(label_lake_run / "label_lake_summary.json"),
            "label_lake_labels_sha256": labels_hash,
            "lifecycle_event_count": len(lifecycle_by_node),
        })
    return summary, all_facts, source_summaries


def _recovery_status_counts(labels: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        status = str(label.get("recovery_status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _build_recovery_label(
    *,
    seq: int,
    run_id: str,
    timestamp_ns: int,
    feature_label: dict[str, Any],
    group_fact: dict[str, Any] | None,
    source_key_count: int,
) -> dict[str, Any]:
    canonical_group_id = str(feature_label.get("canonical_group_id") or "")
    input_horizon = _safe_int(feature_label.get("observed_horizon_source_ticks"))
    outcome_status = str(feature_label.get("outcome_status") or "UNKNOWN")
    recovered_horizon: int | None = None
    if input_horizon is not None:
        recovery_status = "PRESERVED_EXISTING_OBSERVED_HORIZON"
    elif outcome_status == "OBSERVED_FILLED":
        recovery_status = "FILL_HORIZON_REQUIRES_SEPARATE_TIMEBASE"
    elif outcome_status != "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
        recovery_status = "UNRECOVERED_NON_OBSERVED_TERMINAL_OUTCOME"
    elif group_fact is None:
        recovery_status = "UNRECOVERED_MISSING_CANONICAL_LIFECYCLE_GROUP"
    elif group_fact.get("terminal_horizon_source_ticks") is None:
        recovery_status = "UNRECOVERED_MISSING_TERMINAL_SOURCE_TICKS"
    else:
        recovery_status = "RECOVERED_TERMINAL_SOURCE_TICKS"
        recovered_horizon = int(group_fact["terminal_horizon_source_ticks"])
    effective_horizon = input_horizon if input_horizon is not None else recovered_horizon
    return {
        "schema_version": 1,
        "label_type": "PHASE51J_OBSERVED_HORIZON_RECOVERY_LABEL",
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
        "source_order_key_count": source_key_count,
        "venue_id": feature_label.get("venue_id") or "UNKNOWN",
        "side": _canonical_side(feature_label.get("side")),
        "order_holdout_split": str(feature_label.get("order_holdout_split") or "UNKNOWN").upper(),
        "outcome_status": outcome_status,
        "p_fill_outcome": feature_label.get("p_fill_outcome"),
        "input_observed_horizon_available": input_horizon is not None,
        "input_observed_horizon_source_ticks": input_horizon,
        "recovery_status": recovery_status,
        "recovered_observed_horizon_source_ticks": recovered_horizon,
        "effective_observed_horizon_available": effective_horizon is not None,
        "effective_observed_horizon_source_ticks": effective_horizon,
        "group_place_source_t_min": group_fact.get("place_source_t_min") if group_fact else None,
        "group_terminal_source_t_min": group_fact.get("terminal_source_t_min") if group_fact else None,
        "group_terminal_action_counts": group_fact.get("terminal_action_counts") if group_fact else {},
        "group_terminal_event_count": int(group_fact.get("terminal_event_count") or 0) if group_fact else 0,
        "group_lifecycle_event_count": int(group_fact.get("lifecycle_event_count") or 0) if group_fact else 0,
    }


def _empty_bucket_counts() -> dict[str, int]:
    return {
        "label_count": 0,
        "input_observed_horizon_available_count": 0,
        "input_observed_horizon_missing_count": 0,
        "preserved_existing_observed_horizon_count": 0,
        "recovered_terminal_horizon_count": 0,
        "fill_horizon_requires_separate_timebase_count": 0,
        "unrecovered_missing_lifecycle_group_count": 0,
        "unrecovered_missing_terminal_source_ticks_count": 0,
        "effective_observed_horizon_available_count": 0,
        "effective_observed_horizon_missing_count": 0,
    }


def _add_to_bucket(counts: dict[str, int], label: dict[str, Any]) -> None:
    counts["label_count"] += 1
    if label.get("input_observed_horizon_available"):
        counts["input_observed_horizon_available_count"] += 1
    else:
        counts["input_observed_horizon_missing_count"] += 1
    status = label.get("recovery_status")
    if status == "PRESERVED_EXISTING_OBSERVED_HORIZON":
        counts["preserved_existing_observed_horizon_count"] += 1
    elif status == "RECOVERED_TERMINAL_SOURCE_TICKS":
        counts["recovered_terminal_horizon_count"] += 1
    elif status == "FILL_HORIZON_REQUIRES_SEPARATE_TIMEBASE":
        counts["fill_horizon_requires_separate_timebase_count"] += 1
    elif status == "UNRECOVERED_MISSING_CANONICAL_LIFECYCLE_GROUP":
        counts["unrecovered_missing_lifecycle_group_count"] += 1
    elif status == "UNRECOVERED_MISSING_TERMINAL_SOURCE_TICKS":
        counts["unrecovered_missing_terminal_source_ticks_count"] += 1
    if label.get("effective_observed_horizon_available"):
        counts["effective_observed_horizon_available_count"] += 1
    else:
        counts["effective_observed_horizon_missing_count"] += 1


def _bucket_entries(label: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    venue = str(label.get("venue_id") or "UNKNOWN")
    side = str(label.get("side") or "UNKNOWN")
    return [
        ("GLOBAL", {"scope": "GLOBAL"}),
        (f"VENUE:{venue}", {"venue_id": venue}),
        (f"SIDE:{side}", {"side": side}),
        (f"VENUE_SIDE:{venue}:{side}", {"venue_id": venue, "side": side}),
    ]


def _bucket_gate_reasons(counts: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    if counts["recovered_terminal_horizon_count"] > 0:
        reasons.append("recovered_terminal_source_tick_horizons")
    if counts["fill_horizon_requires_separate_timebase_count"] > 0:
        reasons.append("fill_horizon_requires_separate_timebase")
    if (
        counts["unrecovered_missing_lifecycle_group_count"] > 0
        or counts["unrecovered_missing_terminal_source_ticks_count"] > 0
        or counts["effective_observed_horizon_missing_count"] > 0
    ):
        reasons.append("missing_observed_horizon_features")
    reasons.append("requires_phase51h_phase51i_replay_before_model_training")
    return reasons


def _build_bucket_records(labels: list[dict[str, Any]], *, run_id: str, timestamp_ns: int) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for label in labels:
        for bucket_id, dimensions in _bucket_entries(label):
            bucket = buckets.setdefault(bucket_id, {"dimensions": dimensions, "counts": _empty_bucket_counts()})
            _add_to_bucket(bucket["counts"], label)
    records: list[dict[str, Any]] = []
    for seq, (bucket_id, bucket) in enumerate(sorted(buckets.items()), start=1):
        counts = bucket["counts"]
        records.append({
            "schema_version": 1,
            "label_type": "PHASE51J_OBSERVED_HORIZON_RECOVERY_BUCKET",
            "bucket_seq": seq,
            "timestamp_local_ns": timestamp_ns + seq,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "bucket_id": bucket_id,
            "bucket_dimensions": bucket["dimensions"],
            "gate_status": "HOLD",
            "gate_reasons": _bucket_gate_reasons(counts),
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


def _summary_gate_reason(global_bucket: dict[str, Any]) -> str:
    if int(global_bucket.get("effective_observed_horizon_missing_count") or 0) > 0:
        return "phase51j_observed_horizon_recovery_partial_horizon_missing"
    return "phase51j_observed_horizon_recovery_complete_nonlive_hold"


def build_observed_horizon_recovery(
    *,
    feature_audit_run: Path,
    canonical_pfill_run: Path,
    lifecycle_truth_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51J-OBSERVED-HORIZON-RECOVERY-{_utc_stamp()}"
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

    feature_summary, feature_labels = _load_feature_audit(feature_audit_run)
    canonical_summary = _load_canonical_summary(canonical_pfill_run)
    source_keys_by_group = _source_order_keys_by_group(canonical_pfill_run)
    lifecycle_summary, group_facts, lifecycle_source_summaries = _load_lifecycle_timing_facts(lifecycle_truth_run)

    recovery_labels: list[dict[str, Any]] = []
    for seq, feature_label in enumerate(feature_labels, start=1):
        canonical_group_id = str(feature_label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError("feature audit label missing canonical_group_id")
        if canonical_group_id not in source_keys_by_group:
            raise ValueError(f"canonical_group_id {canonical_group_id} missing from canonical source manifest")
        recovery_labels.append(_build_recovery_label(
            seq=seq,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            feature_label=feature_label,
            group_fact=group_facts.get(canonical_group_id),
            source_key_count=len(source_keys_by_group.get(canonical_group_id) or []),
        ))

    for label in recovery_labels:
        _check_no_raw_identifiers(label, out_dir / "observed_horizon_recovery_labels.jsonl", label="recovery label")

    bucket_records = _build_bucket_records(recovery_labels, run_id=run_id, timestamp_ns=timestamp_ns)
    global_bucket = next(record for record in bucket_records if record["bucket_id"] == "GLOBAL")
    status_counts = _recovery_status_counts(recovery_labels)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": _summary_gate_reason(global_bucket),
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
        "canonical_source_manifest_sha256": _sha256_file(canonical_pfill_run / "source_to_canonical_order_manifest.jsonl"),
        "lifecycle_truth_summary_sha256": _sha256_file(lifecycle_truth_run / "lifecycle_truth_audit_summary.json"),
        "lifecycle_source_summaries": lifecycle_source_summaries,
        "source_telemetry_sha256_list": feature_summary.get("source_telemetry_sha256_list") or [],
        "input_feature_gate_reason": feature_summary.get("gate_reason"),
        "input_lifecycle_gate_reason": lifecycle_summary.get("gate_reason"),
        "recovery_status_counts": status_counts,
        "bucket_count": len(bucket_records),
        **{
            key: global_bucket[key]
            for key in (
                "label_count",
                "input_observed_horizon_available_count",
                "input_observed_horizon_missing_count",
                "preserved_existing_observed_horizon_count",
                "recovered_terminal_horizon_count",
                "fill_horizon_requires_separate_timebase_count",
                "unrecovered_missing_lifecycle_group_count",
                "unrecovered_missing_terminal_source_ticks_count",
                "effective_observed_horizon_available_count",
                "effective_observed_horizon_missing_count",
            )
        },
    }

    labels_path = out_dir / "observed_horizon_recovery_labels.jsonl"
    buckets_path = out_dir / "observed_horizon_recovery_buckets.jsonl"
    summary_path = out_dir / "observed_horizon_recovery_summary.json"
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
        out_dir = build_observed_horizon_recovery(
            feature_audit_run=args.feature_audit_run,
            canonical_pfill_run=args.canonical_pfill_run,
            lifecycle_truth_run=args.lifecycle_truth_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51j_observed_horizon_recovery: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51j_observed_horizon_recovery: wrote {out_dir}")
    print("phase51j_observed_horizon_recovery: status HOLD (horizon recovery evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
