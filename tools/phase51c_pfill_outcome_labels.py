#!/usr/bin/env python3
"""Build Phase 5.1c order-level P_fill outcome labels.

This tool is an offline evidence gate. It consumes an existing Phase 5.1c
label lake and deterministic join/holdout pack, then emits one outcome label
per observed place order. It does not train a model, submit orders, authorize
live/canary use, or create financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_pfill_outcome"
TERMINAL_ACTIONS = {"cancel", "replace", "cancel_all", "expire", "expired", "reject", "rejected"}


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
        "admissible_for_model_training": False,
    }


def _holdout_split(stable_key: str) -> str:
    return "HOLDOUT" if int(hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 else "TRAIN"


def _order_identity(order: dict[str, Any]) -> str:
    return _stable_hash([
        order.get("label_seq"),
        order.get("source_line"),
        order.get("source_t"),
        order.get("source_order_index"),
        order.get("venue_id"),
        order.get("side"),
        order.get("price"),
        order.get("size"),
        order.get("decision_id"),
        order.get("order_id_hash"),
        order.get("client_order_id_hash"),
    ])


def _candidate_keys_from_order(order: dict[str, Any], *, include_decision_id: bool = False) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if order.get("label_seq") is not None:
        keys.append(("label_seq", str(order.get("label_seq"))))
    source_tuple = (order.get("source_line"), order.get("source_t"), order.get("source_order_index"))
    if all(value is not None for value in source_tuple):
        keys.append(("source", _stable_hash(source_tuple)))
    if order.get("order_id_hash"):
        keys.append(("order_id_hash", str(order.get("order_id_hash"))))
    if order.get("client_order_id_hash"):
        keys.append(("client_order_id_hash", str(order.get("client_order_id_hash"))))
    if include_decision_id and order.get("decision_id"):
        keys.append(("decision_id", str(order.get("decision_id"))))
    return keys


def _candidate_keys_from_join(label: dict[str, Any], *, include_decision_id: bool = False) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if label.get("order_label_seq") is not None:
        keys.append(("label_seq", str(label.get("order_label_seq"))))
    source_tuple = (label.get("order_source_line"), label.get("order_source_t"), label.get("order_source_order_index"))
    if all(value is not None for value in source_tuple):
        keys.append(("source", _stable_hash(source_tuple)))
    if label.get("order_id_hash"):
        keys.append(("order_id_hash", str(label.get("order_id_hash"))))
    if label.get("client_order_id_hash"):
        keys.append(("client_order_id_hash", str(label.get("client_order_id_hash"))))
    if include_decision_id and label.get("order_decision_id"):
        keys.append(("decision_id", str(label.get("order_decision_id"))))
    if include_decision_id and label.get("fill_decision_id"):
        keys.append(("decision_id", str(label.get("fill_decision_id"))))
    return keys


def _register_key(index: dict[tuple[str, str], str | None], key: tuple[str, str], order_key: str) -> None:
    if key in index and index[key] != order_key:
        index[key] = None
    else:
        index[key] = order_key


def _resolve_order(index: dict[tuple[str, str], str | None], keys: list[tuple[str, str]]) -> tuple[str | None, str]:
    for key in keys:
        if key not in index:
            continue
        order_key = index[key]
        if order_key is None:
            return None, f"ambiguous_{key[0]}"
        return order_key, key[0]
    return None, "unmatched"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_label_lake(label_lake_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[tuple[str, str], str | None], dict[str, list[dict[str, Any]]]]:
    summary = _load_json(label_lake_run / "label_lake_summary.json")
    place_orders: dict[str, dict[str, Any]] = {}
    key_index: dict[tuple[str, str], str | None] = {}
    terminal_events_by_order: dict[str, list[dict[str, Any]]] = {}
    non_place_events: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(label_lake_run / "labels.jsonl"):
        if label.get("label_type") != "ORDER_LIFECYCLE_LABEL":
            continue
        action = str(label.get("action") or "").lower()
        if action == "place":
            order_key = _order_identity(label)
            place_orders[order_key] = label
            for key in _candidate_keys_from_order(label):
                _register_key(key_index, key, order_key)
        else:
            non_place_events.append(label)
    for event in non_place_events:
        action = str(event.get("action") or "").lower()
        if action not in TERMINAL_ACTIONS:
            continue
        order_key, _ = _resolve_order(key_index, _candidate_keys_from_order(event))
        if order_key:
            terminal_events_by_order.setdefault(order_key, []).append(event)
    return summary, place_orders, key_index, terminal_events_by_order


def _load_join_fills(join_holdout_run: Path, key_index: dict[tuple[str, str], str | None]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], int]:
    fills_by_order: dict[str, list[dict[str, Any]]] = {}
    reason_counts: dict[str, int] = {}
    fill_labels_seen = 0
    for _, label in _iter_jsonl(join_holdout_run / "joined_labels.jsonl"):
        if label.get("label_type") != "DETERMINISTIC_JOIN_LABEL":
            continue
        fill_labels_seen += 1
        order_key, reason = _resolve_order(key_index, _candidate_keys_from_join(label))
        if order_key:
            fills_by_order.setdefault(order_key, []).append(label)
        else:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return fills_by_order, reason_counts, fill_labels_seen


def _aggregate_fills(fills: list[dict[str, Any]]) -> dict[str, Any]:
    fill_times = [int(value) for value in (fill.get("fill_time_ms") for fill in fills) if value is not None]
    sizes = [_safe_float(fill.get("size")) for fill in fills]
    valid_sizes = [size for size in sizes if size is not None]
    return {
        "fill_count": len(fills),
        "first_fill_time_ms": min(fill_times) if fill_times else None,
        "last_fill_time_ms": max(fill_times) if fill_times else None,
        "filled_size_total": sum(valid_sizes) if valid_sizes else None,
        "maker_taker_role_counts": {
            "MAKER": sum(1 for fill in fills if fill.get("maker_taker_role") == "MAKER"),
            "TAKER": sum(1 for fill in fills if fill.get("maker_taker_role") == "TAKER"),
            "UNKNOWN": sum(1 for fill in fills if fill.get("maker_taker_role") == "UNKNOWN"),
        },
    }


def _terminal_summary(order: dict[str, Any], terminal_events: list[dict[str, Any]]) -> dict[str, Any]:
    if not terminal_events:
        return {
            "terminal_event_count": 0,
            "terminal_action_first": None,
            "terminal_source_line_first": None,
            "terminal_source_t_first": None,
            "observed_horizon_source_ticks": None,
        }
    ordered = sorted(terminal_events, key=lambda event: (
        event.get("source_line") if event.get("source_line") is not None else 10**18,
        event.get("source_order_index") if event.get("source_order_index") is not None else 10**18,
    ))
    first = ordered[0]
    horizon = None
    if order.get("source_t") is not None and first.get("source_t") is not None:
        try:
            horizon = int(first.get("source_t")) - int(order.get("source_t"))
        except (TypeError, ValueError):
            horizon = None
    return {
        "terminal_event_count": len(terminal_events),
        "terminal_action_first": first.get("action"),
        "terminal_source_line_first": first.get("source_line"),
        "terminal_source_t_first": first.get("source_t"),
        "observed_horizon_source_ticks": horizon,
    }


def _gate_reason(counts: dict[str, int]) -> str:
    if counts["order_label_count"] == 0:
        return "pfill_outcome_missing_place_orders"
    if counts["filled_count"] == 0:
        return "pfill_outcome_missing_observed_fills"
    if counts["filled_count"] < 200:
        return "pfill_outcome_sparse_observed_fills"
    if counts["holdout_count"] == 0:
        return "pfill_outcome_missing_holdout"
    if counts["censored_count"] > 0:
        return "pfill_outcome_contains_censored_orders"
    return "pfill_outcome_requires_board_review"


def build_pfill_outcomes(
    *,
    label_lake_run: Path,
    join_holdout_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51C-PFILL-OUTCOME-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    lake_summary, place_orders, key_index, terminal_events_by_order = _load_label_lake(label_lake_run)
    join_summary = _load_json(join_holdout_run / "join_holdout_summary.json")
    if lake_summary.get("source_telemetry_sha256") != join_summary.get("source_telemetry_sha256"):
        raise ValueError("label lake and join holdout must share source_telemetry_sha256")
    fills_by_order, unresolved_fill_reason_counts, fill_labels_seen = _load_join_fills(join_holdout_run, key_index)

    labels: list[dict[str, Any]] = []
    counts = {
        "order_label_count": 0,
        "filled_count": 0,
        "not_filled_count": 0,
        "censored_count": 0,
        "train_count": 0,
        "holdout_count": 0,
    }
    action_counts: dict[str, int] = {}
    for seq, (order_key, order) in enumerate(sorted(place_orders.items(), key=lambda item: item[0]), start=1):
        fills = fills_by_order.get(order_key, [])
        terminal_events = terminal_events_by_order.get(order_key, [])
        fill_summary = _aggregate_fills(fills)
        terminal = _terminal_summary(order, terminal_events)
        if fills:
            outcome = 1.0
            outcome_status = "OBSERVED_FILLED"
            counts["filled_count"] += 1
        elif terminal_events:
            outcome = 0.0
            outcome_status = "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL"
            counts["not_filled_count"] += 1
        else:
            outcome = None
            outcome_status = "CENSORED_OR_UNOBSERVED"
            counts["censored_count"] += 1
        split = _holdout_split(order_key)
        counts["holdout_count" if split == "HOLDOUT" else "train_count"] += 1
        counts["order_label_count"] += 1
        action_counts[str(order.get("action") or "UNKNOWN")] = action_counts.get(str(order.get("action") or "UNKNOWN"), 0) + 1
        label = _base_label(run_id, seq, timestamp_ns)
        label.update({
            "source": "phase51c_label_lake_and_join_holdout",
            "source_telemetry_sha256": lake_summary.get("source_telemetry_sha256"),
            "label_lake_run": str(label_lake_run),
            "join_holdout_run": str(join_holdout_run),
            "order_key": order_key,
            "order_label_seq": order.get("label_seq"),
            "order_source_line": order.get("source_line"),
            "order_source_t": order.get("source_t"),
            "order_source_order_index": order.get("source_order_index"),
            "venue_id": order.get("venue_id"),
            "side": order.get("side"),
            "price": order.get("price"),
            "size": order.get("size"),
            "decision_id": order.get("decision_id"),
            "order_id_hash": order.get("order_id_hash"),
            "client_order_id_hash": order.get("client_order_id_hash"),
            "order_holdout_split": split,
            "p_fill_outcome": outcome,
            "outcome_status": outcome_status,
            "label_status": "OBSERVED_ORDER_PFILL_OUTCOME" if outcome is not None else "CENSORED_ORDER_PFILL_OUTCOME",
            "training_hold_reason": "requires_pfill_model_calibration_and_board_review",
            **fill_summary,
            **terminal,
        })
        labels.append(label)

    gate_reason = _gate_reason(counts)
    labels_path = out_dir / "pfill_order_labels.jsonl"
    summary_path = out_dir / "pfill_outcome_summary.json"
    _write_jsonl(labels_path, labels)
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
        "label_lake_run": str(label_lake_run),
        "join_holdout_run": str(join_holdout_run),
        "source_telemetry_sha256": lake_summary.get("source_telemetry_sha256"),
        "label_lake_summary_sha256": _sha256_file(label_lake_run / "label_lake_summary.json"),
        "join_holdout_summary_sha256": _sha256_file(join_holdout_run / "join_holdout_summary.json"),
        "fill_labels_seen": fill_labels_seen,
        "unresolved_fill_reason_counts": unresolved_fill_reason_counts,
        "order_action_counts": action_counts,
        "p_fill_positive_rate_observed": (
            counts["filled_count"] / (counts["filled_count"] + counts["not_filled_count"])
            if counts["filled_count"] + counts["not_filled_count"] > 0
            else None
        ),
        **counts,
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-lake-run", type=Path, required=True)
    parser.add_argument("--join-holdout-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_pfill_outcomes(
            label_lake_run=args.label_lake_run,
            join_holdout_run=args.join_holdout_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51c_pfill_outcome_labels: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_pfill_outcome_labels: wrote {out_dir}")
    print("phase51c_pfill_outcome_labels: status HOLD (order-level P_fill evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
