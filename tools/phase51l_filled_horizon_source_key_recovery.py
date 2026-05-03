#!/usr/bin/env python3
"""Recover Phase 5.1 filled-order horizons via source keys and hashed fills.

This is an offline HOLD-only evidence gate. It consumes the Phase 5.1k
timebase recovery pack, canonical P_fill outcomes, source P_fill outcome packs,
and observed fill-label packs. It only writes source-tick horizons when they can
be reconstructed from source-tick evidence; raw order/fill identifiers are never
emitted.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51l_filled_horizon_source_key_recovery"

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
    "source_decision_id",
}

HASH_FIELDS = ("order_id_hash", "client_order_id_hash")


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


def _count_map(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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


def _load_phase51k(
    phase51k_recovery_run: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summary = _load_hold_summary(phase51k_recovery_run, "filled_horizon_timebase_recovery_summary.json")
    if summary.get("raw_identifier_redaction_status") != "PASS":
        raise ValueError(f"{phase51k_recovery_run} must have raw_identifier_redaction_status=PASS")
    labels_path = phase51k_recovery_run / "filled_horizon_timebase_recovery_labels.jsonl"
    labels: list[dict[str, Any]] = []
    by_group: dict[str, dict[str, Any]] = {}
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "PHASE51K_FILLED_HORIZON_TIMEBASE_RECOVERY_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        _check_no_raw_identifiers(label, labels_path, label="label")
        if label.get("raw_identifier_redaction_status") != "PASS":
            raise ValueError(f"{labels_path} contains non-PASS raw identifier redaction status")
        canonical_group_id = str(label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError(f"{labels_path} row missing canonical_group_id")
        if canonical_group_id in by_group:
            raise ValueError(f"{labels_path} duplicate canonical_group_id={canonical_group_id}")
        by_group[canonical_group_id] = label
        labels.append(label)
    expected = int(summary.get("label_count") or 0)
    if len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary label_count {expected}")
    return summary, by_group, labels


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


def _load_source_pfill_runs(
    source_pfill_runs: list[Path],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    by_source_order_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    seen_runs: set[Path] = set()
    for run in source_pfill_runs:
        run = _resolve_path(run)
        assert run is not None
        if run in seen_runs:
            continue
        seen_runs.add(run)
        summary = _load_hold_summary(run, "pfill_outcome_summary.json")
        source_sha = str(summary.get("source_telemetry_sha256") or "")
        if not source_sha:
            raise ValueError(f"{run} pfill summary missing source_telemetry_sha256")
        labels_path = run / "pfill_order_labels.jsonl"
        label_count = 0
        for _, label in _iter_jsonl(labels_path):
            if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
                continue
            _check_unsafe(label, labels_path, label="label")
            label_source_sha = str(label.get("source_telemetry_sha256") or source_sha)
            if label_source_sha != source_sha:
                raise ValueError(f"{labels_path} source_telemetry_sha256 mismatch")
            order_key = str(label.get("order_key") or "")
            if not order_key:
                raise ValueError(f"{labels_path} row missing order_key")
            by_source_order_key.setdefault((source_sha, order_key), []).append(label)
            label_count += 1
        expected = int(summary.get("order_label_count") or label_count)
        if label_count != expected:
            raise ValueError(f"{labels_path} label count {label_count} != summary order_label_count {expected}")
        summaries.append({
            "run_id": summary.get("run_id"),
            "run_path": str(run),
            "source_telemetry_sha256": source_sha,
            "gate_reason": summary.get("gate_reason"),
            "gate_status": summary.get("gate_status"),
            "pfill_outcome_summary_sha256": _sha256_file(run / "pfill_outcome_summary.json"),
            "pfill_order_labels_sha256": _sha256_file(labels_path),
            "order_label_count": label_count,
        })
    return by_source_order_key, summaries


def _load_observed_label_runs(
    observed_label_runs: list[Path],
) -> tuple[dict[tuple[str, str, str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    by_hash: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    seen_runs: set[Path] = set()
    for run in observed_label_runs:
        run = _resolve_path(run)
        assert run is not None
        if run in seen_runs:
            continue
        seen_runs.add(run)
        summary = _load_hold_summary(run, "observed_label_summary.json")
        source_sha = str(summary.get("source_telemetry_sha256") or "")
        if not source_sha:
            raise ValueError(f"{run} observed summary missing source_telemetry_sha256")
        labels_path = run / "labels.jsonl"
        fill_label_count = 0
        for _, label in _iter_jsonl(labels_path):
            if label.get("label_type") != "OBSERVED_FILL_LABEL":
                continue
            _check_unsafe(label, labels_path, label="observed fill label")
            label_source_sha = str(label.get("source_telemetry_sha256") or source_sha)
            venue = str(label.get("venue_id") or "UNKNOWN")
            for field in HASH_FIELDS:
                hashed_value = str(label.get(field) or "")
                if hashed_value:
                    by_hash.setdefault((label_source_sha, venue, field, hashed_value), []).append(label)
            fill_label_count += 1
        summaries.append({
            "run_id": summary.get("run_id"),
            "run_path": str(run),
            "source_telemetry_sha256": source_sha,
            "gate_reason": summary.get("gate_reason"),
            "gate_status": summary.get("gate_status"),
            "observed_label_summary_sha256": _sha256_file(run / "observed_label_summary.json"),
            "observed_labels_sha256": _sha256_file(labels_path),
            "fill_labels": fill_label_count,
        })
    return by_hash, summaries


def _source_order_keys(canonical_label: dict[str, Any] | None, phase51k_label: dict[str, Any]) -> list[str]:
    keys = [
        str(value)
        for value in ((canonical_label or {}).get("source_order_keys") or [])
        if value
    ]
    if not keys and (canonical_label or {}).get("order_key"):
        keys = [str((canonical_label or {})["order_key"])]
    if not keys and phase51k_label.get("canonical_order_key"):
        keys = [str(phase51k_label["canonical_order_key"])]
    return sorted(set(keys))


def _candidate_hashes(*labels: dict[str, Any] | None) -> dict[str, list[str]]:
    by_field: dict[str, list[str]] = {field: [] for field in HASH_FIELDS}
    for label in labels:
        if label is None:
            continue
        for field in HASH_FIELDS:
            value = str(label.get(field) or "")
            if value and value not in by_field[field]:
                by_field[field].append(value)
    return by_field


def _pick_earliest_nonnegative_horizon(candidates: list[dict[str, int]]) -> tuple[int | None, dict[str, Any]]:
    nonnegative = [candidate for candidate in candidates if candidate["horizon"] >= 0]
    if not candidates:
        return None, {"candidate_count": 0, "nonnegative_candidate_count": 0}
    if not nonnegative:
        return None, {
            "candidate_count": len(candidates),
            "nonnegative_candidate_count": 0,
            "negative_horizon_count": len(candidates),
            "earliest_candidate_source_t": min(candidate["fill_source_t"] for candidate in candidates),
        }
    selected = sorted(nonnegative, key=lambda item: (item["horizon"], item["fill_source_t"]))[0]
    return selected["horizon"], {
        "candidate_count": len(candidates),
        "nonnegative_candidate_count": len(nonnegative),
        "negative_horizon_count": len(candidates) - len(nonnegative),
        "earliest_candidate_source_t": min(candidate["fill_source_t"] for candidate in candidates),
        "selected_candidate_source_t": selected["fill_source_t"],
    }


def _recover_from_source_pfill(
    *,
    canonical_order_source_t: int,
    source_sha: str,
    source_order_keys: list[str],
    source_pfill_by_order_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[str, int | None, dict[str, Any], list[dict[str, Any]]]:
    matched_labels: list[dict[str, Any]] = []
    filled_labels: list[dict[str, Any]] = []
    candidates: list[dict[str, int]] = []
    for source_order_key in source_order_keys:
        labels = source_pfill_by_order_key.get((source_sha, source_order_key), [])
        matched_labels.extend(labels)
        for label in labels:
            if label.get("outcome_status") != "OBSERVED_FILLED":
                continue
            filled_labels.append(label)
            source_order_t = _safe_int(label.get("order_source_t"))
            observed_horizon = _safe_int(label.get("observed_horizon_source_ticks"))
            if source_order_t is None or observed_horizon is None:
                continue
            fill_source_t = source_order_t + observed_horizon
            candidates.append({
                "fill_source_t": fill_source_t,
                "horizon": fill_source_t - canonical_order_source_t,
            })
    detail = {
        "source_order_key_count": len(source_order_keys),
        "candidate_source_pfill_label_count": len(matched_labels),
        "candidate_source_filled_label_count": len(filled_labels),
        "candidate_source_pfill_horizon_count": len(candidates),
    }
    if not matched_labels:
        return "NO_SOURCE_PFILL_LABEL", None, detail, matched_labels
    if not filled_labels:
        return "NO_SOURCE_FILLED_LABEL", None, detail, matched_labels
    if not candidates:
        return "SOURCE_FILLED_HORIZON_MISSING", None, detail, matched_labels
    value, pick_detail = _pick_earliest_nonnegative_horizon(candidates)
    detail.update(pick_detail)
    if value is None:
        return "NEGATIVE_HORIZON", None, detail, matched_labels
    return "RECOVERED_FROM_SOURCE_PFILL_HORIZON", value, detail, matched_labels


def _recover_from_observed_hash(
    *,
    canonical_order_source_t: int,
    source_sha: str,
    venue_id: str,
    hashes: dict[str, list[str]],
    observed_by_hash: dict[tuple[str, str, str, str], list[dict[str, Any]]],
) -> tuple[str, int | None, dict[str, Any]]:
    candidates: list[dict[str, int]] = []
    hash_count = 0
    matched_fill_count = 0
    for field, values in hashes.items():
        for hashed_value in values:
            hash_count += 1
            labels = observed_by_hash.get((source_sha, venue_id, field, hashed_value), [])
            matched_fill_count += len(labels)
            for label in labels:
                source_t = _safe_int(label.get("source_t"))
                if source_t is None:
                    continue
                candidates.append({
                    "fill_source_t": source_t,
                    "horizon": source_t - canonical_order_source_t,
                })
    detail = {
        "candidate_hash_value_count": hash_count,
        "candidate_observed_fill_hash_count": matched_fill_count,
        "candidate_observed_fill_source_tick_count": len(candidates),
    }
    if hash_count == 0:
        return "OBSERVED_FILL_HASH_MISSING", None, detail
    if matched_fill_count == 0:
        return "NO_OBSERVED_FILL_HASH_MATCH", None, detail
    if not candidates:
        return "OBSERVED_FILL_SOURCE_TICK_MISSING", None, detail
    value, pick_detail = _pick_earliest_nonnegative_horizon(candidates)
    detail.update(pick_detail)
    if value is None:
        return "NEGATIVE_HORIZON", None, detail
    return "RECOVERED_FROM_OBSERVED_FILL_HASH", value, detail


def _build_recovery_label(
    *,
    seq: int,
    run_id: str,
    timestamp_ns: int,
    phase51k_label: dict[str, Any],
    canonical_label: dict[str, Any] | None,
    source_pfill_by_order_key: dict[tuple[str, str], list[dict[str, Any]]],
    observed_by_hash: dict[tuple[str, str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    canonical_group_id = str(phase51k_label.get("canonical_group_id") or "")
    source_sha = str(phase51k_label.get("source_telemetry_sha256") or "")
    venue_id = str(phase51k_label.get("venue_id") or "UNKNOWN")
    outcome_status = str(phase51k_label.get("outcome_status") or "UNKNOWN")
    upstream_status = str(phase51k_label.get("recovery_status") or "UNKNOWN")
    input_horizon = _safe_int(phase51k_label.get("effective_observed_horizon_source_ticks"))
    canonical_order_source_t = _first_int(
        (canonical_label or {}).get("order_source_t"),
        phase51k_label.get("order_source_t"),
    )
    source_order_keys = _source_order_keys(canonical_label, phase51k_label)
    recovered_horizon: int | None = None
    recovery_method = "NONE"
    source_detail: dict[str, Any] = {}
    hash_detail: dict[str, Any] = {}
    matched_source_labels: list[dict[str, Any]] = []

    if input_horizon is not None:
        recovery_status = "PRESERVED_EXISTING_SOURCE_TICKS"
        recovered_horizon = input_horizon
        recovery_method = "UPSTREAM_PHASE51K"
    elif outcome_status != "OBSERVED_FILLED":
        recovery_status = "NOT_FILLED_NOT_APPLICABLE"
    elif upstream_status != "MISSING_JOIN":
        recovery_status = "UPSTREAM_UNRECOVERED_NOT_TARGETED"
    elif canonical_label is None:
        recovery_status = "MISSING_CANONICAL_LABEL"
    elif canonical_order_source_t is None:
        recovery_status = "MISSING_CANONICAL_SOURCE_TICK"
    elif not source_order_keys:
        recovery_status = "MISSING_SOURCE_ORDER_KEYS"
    else:
        source_status, source_value, source_detail, matched_source_labels = _recover_from_source_pfill(
            canonical_order_source_t=canonical_order_source_t,
            source_sha=source_sha,
            source_order_keys=source_order_keys,
            source_pfill_by_order_key=source_pfill_by_order_key,
        )
        if source_value is not None:
            recovery_status = source_status
            recovered_horizon = source_value
            recovery_method = "SOURCE_PFILL_HORIZON"
        else:
            hashes = _candidate_hashes(canonical_label, *matched_source_labels)
            hash_status, hash_value, hash_detail = _recover_from_observed_hash(
                canonical_order_source_t=canonical_order_source_t,
                source_sha=source_sha,
                venue_id=venue_id,
                hashes=hashes,
                observed_by_hash=observed_by_hash,
            )
            if hash_value is not None:
                recovery_status = hash_status
                recovered_horizon = hash_value
                recovery_method = "OBSERVED_FILL_HASH"
            else:
                recovery_status = hash_status if hash_status != "OBSERVED_FILL_HASH_MISSING" else source_status
                recovery_method = "UNRECOVERED"

    effective_horizon = input_horizon if input_horizon is not None else recovered_horizon
    return {
        "schema_version": 1,
        "label_type": "PHASE51L_FILLED_HORIZON_SOURCE_KEY_RECOVERY_LABEL",
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
        "canonical_order_key": phase51k_label.get("canonical_order_key"),
        "source_telemetry_sha256": source_sha,
        "venue_id": venue_id,
        "side": _canonical_side(phase51k_label.get("side")),
        "order_holdout_split": str(phase51k_label.get("order_holdout_split") or "UNKNOWN").upper(),
        "outcome_status": outcome_status,
        "p_fill_outcome": phase51k_label.get("p_fill_outcome"),
        "fill_count": _safe_int(phase51k_label.get("fill_count")) or 0,
        "upstream_recovery_status": upstream_status,
        "input_observed_horizon_source_ticks": input_horizon,
        "canonical_order_source_t_available": canonical_order_source_t is not None,
        "source_order_key_count": len(source_order_keys),
        "recovery_status": recovery_status,
        "recovery_method": recovery_method,
        "recovery_timebase": "SOURCE_TICKS" if effective_horizon is not None else "NONE",
        "recovered_observed_horizon_source_ticks": (
            recovered_horizon if input_horizon is None else None
        ),
        "effective_observed_horizon_source_ticks": effective_horizon,
        "effective_observed_horizon_source_ticks_available": effective_horizon is not None,
        "candidate_source_pfill_label_count": int(source_detail.get("candidate_source_pfill_label_count") or 0),
        "candidate_source_filled_label_count": int(source_detail.get("candidate_source_filled_label_count") or 0),
        "candidate_source_pfill_horizon_count": int(source_detail.get("candidate_source_pfill_horizon_count") or 0),
        "candidate_hash_value_count": int(hash_detail.get("candidate_hash_value_count") or 0),
        "candidate_observed_fill_hash_count": int(hash_detail.get("candidate_observed_fill_hash_count") or 0),
        "candidate_observed_fill_source_tick_count": int(
            hash_detail.get("candidate_observed_fill_source_tick_count") or 0
        ),
        "source_negative_horizon_count": int(source_detail.get("negative_horizon_count") or 0),
        "hash_negative_horizon_count": int(hash_detail.get("negative_horizon_count") or 0),
        "source_selected_candidate_source_t": source_detail.get("selected_candidate_source_t"),
        "hash_selected_candidate_source_t": hash_detail.get("selected_candidate_source_t"),
        "timebase_isolated": True,
        "exchange_ms_not_written_to_source_ticks": True,
    }


def _empty_bucket_counts() -> dict[str, int]:
    return {
        "label_count": 0,
        "filled_label_count": 0,
        "target_missing_join_count": 0,
        "preserved_existing_source_tick_count": 0,
        "source_pfill_horizon_recovered_count": 0,
        "observed_fill_hash_recovered_count": 0,
        "recovered_source_tick_count": 0,
        "still_missing_filled_horizon_count": 0,
        "not_filled_not_applicable_count": 0,
        "upstream_unrecovered_not_targeted_count": 0,
        "negative_horizon_count": 0,
        "unresolved_count": 0,
    }


def _add_to_bucket(counts: dict[str, int], label: dict[str, Any]) -> None:
    counts["label_count"] += 1
    status = str(label.get("recovery_status") or "UNKNOWN")
    is_filled = label.get("outcome_status") == "OBSERVED_FILLED"
    if is_filled:
        counts["filled_label_count"] += 1
        if label.get("upstream_recovery_status") == "MISSING_JOIN":
            counts["target_missing_join_count"] += 1
    if status == "PRESERVED_EXISTING_SOURCE_TICKS":
        counts["preserved_existing_source_tick_count"] += 1
    elif status == "RECOVERED_FROM_SOURCE_PFILL_HORIZON":
        counts["source_pfill_horizon_recovered_count"] += 1
        counts["recovered_source_tick_count"] += 1
    elif status == "RECOVERED_FROM_OBSERVED_FILL_HASH":
        counts["observed_fill_hash_recovered_count"] += 1
        counts["recovered_source_tick_count"] += 1
    elif status == "NOT_FILLED_NOT_APPLICABLE":
        counts["not_filled_not_applicable_count"] += 1
    elif status == "UPSTREAM_UNRECOVERED_NOT_TARGETED":
        counts["upstream_unrecovered_not_targeted_count"] += 1
    elif status == "NEGATIVE_HORIZON":
        counts["negative_horizon_count"] += 1
    if is_filled and label.get("effective_observed_horizon_source_ticks") is None:
        counts["still_missing_filled_horizon_count"] += 1
        counts["unresolved_count"] += 1


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
            reasons.insert(0, "filled_horizon_source_key_still_missing")
        records.append({
            "schema_version": 1,
            "label_type": "PHASE51L_FILLED_HORIZON_SOURCE_KEY_RECOVERY_BUCKET",
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
        return "phase51l_filled_horizon_source_key_partial"
    return "phase51l_filled_horizon_source_key_complete_nonlive_hold"


def build_filled_horizon_source_key_recovery(
    *,
    phase51k_recovery_run: Path,
    canonical_pfill_run: Path,
    source_pfill_runs: list[Path],
    observed_label_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51L-FILLED-HORIZON-SOURCE-KEY-RECOVERY-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    phase51k_recovery_run = _resolve_path(phase51k_recovery_run)
    canonical_pfill_run = _resolve_path(canonical_pfill_run)
    assert phase51k_recovery_run is not None and canonical_pfill_run is not None
    source_pfill_runs = [path for path in (_resolve_path(run) for run in source_pfill_runs) if path is not None]
    observed_label_runs = [path for path in (_resolve_path(run) for run in observed_label_runs) if path is not None]
    if not source_pfill_runs:
        raise ValueError("at least one --source-pfill-run is required")
    if not observed_label_runs:
        raise ValueError("at least one --observed-label-run is required")

    phase51k_summary, _, phase51k_labels = _load_phase51k(phase51k_recovery_run)
    canonical_summary, canonical_by_group = _load_canonical_pfill(canonical_pfill_run)
    source_pfill_by_order_key, source_pfill_summaries = _load_source_pfill_runs(source_pfill_runs)
    observed_by_hash, observed_summaries = _load_observed_label_runs(observed_label_runs)

    recovery_labels: list[dict[str, Any]] = []
    for seq, phase51k_label in enumerate(phase51k_labels, start=1):
        canonical_group_id = str(phase51k_label.get("canonical_group_id") or "")
        label = _build_recovery_label(
            seq=seq,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            phase51k_label=phase51k_label,
            canonical_label=canonical_by_group.get(canonical_group_id),
            source_pfill_by_order_key=source_pfill_by_order_key,
            observed_by_hash=observed_by_hash,
        )
        _check_no_raw_identifiers(label, out_dir / "filled_horizon_source_key_recovery_labels.jsonl", label="label")
        recovery_labels.append(label)

    bucket_records = _build_bucket_records(recovery_labels, run_id=run_id, timestamp_ns=timestamp_ns)
    global_bucket = next(record for record in bucket_records if record["bucket_id"] == "GLOBAL")
    global_counts = {key: int(global_bucket.get(key) or 0) for key in _empty_bucket_counts()}
    status_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for label in recovery_labels:
        status = str(label.get("recovery_status") or "UNKNOWN")
        method = str(label.get("recovery_method") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1

    labels_path = out_dir / "filled_horizon_source_key_recovery_labels.jsonl"
    buckets_path = out_dir / "filled_horizon_source_key_recovery_buckets.jsonl"
    summary_path = out_dir / "filled_horizon_source_key_recovery_summary.json"
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
        "phase51k_recovery_run": str(phase51k_recovery_run),
        "canonical_pfill_run": str(canonical_pfill_run),
        "source_pfill_runs": source_pfill_summaries,
        "observed_label_runs": observed_summaries,
        "phase51k_recovery_summary_sha256": _sha256_file(
            phase51k_recovery_run / "filled_horizon_timebase_recovery_summary.json"
        ),
        "phase51k_recovery_labels_sha256": _sha256_file(
            phase51k_recovery_run / "filled_horizon_timebase_recovery_labels.jsonl"
        ),
        "canonical_pfill_summary_hash": _stable_hash(canonical_summary),
        "phase51k_summary_hash": _stable_hash(phase51k_summary),
        "source_telemetry_sha256_list": sorted({
            str(item.get("source_telemetry_sha256") or "")
            for item in source_pfill_summaries + observed_summaries
            if item.get("source_telemetry_sha256")
        }),
        "label_count": len(recovery_labels),
        "bucket_count": len(bucket_records),
        "recovery_status_counts": dict(sorted(status_counts.items())),
        "recovery_method_counts": dict(sorted(method_counts.items())),
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
    parser.add_argument("--phase51k-recovery-run", type=Path, required=True)
    parser.add_argument("--canonical-pfill-run", type=Path, required=True)
    parser.add_argument("--source-pfill-run", type=Path, action="append", default=[])
    parser.add_argument("--observed-label-run", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_filled_horizon_source_key_recovery(
            phase51k_recovery_run=args.phase51k_recovery_run,
            canonical_pfill_run=args.canonical_pfill_run,
            source_pfill_runs=args.source_pfill_run,
            observed_label_runs=args.observed_label_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51l_filled_horizon_source_key_recovery: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51l_filled_horizon_source_key_recovery: wrote {out_dir}")
    print("phase51l_filled_horizon_source_key_recovery: status HOLD (filled horizon evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
