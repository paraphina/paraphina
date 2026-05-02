#!/usr/bin/env python3
"""Build Phase 5.1c queue/churn/native-limit-pressure labels.

This offline evidence gate consumes a Phase 5.1c label lake and P_fill outcome
pack, then emits one order-level queue/churn proxy label per P_fill order. It
does not calibrate a queue model, submit orders, approve live/canary use, or
make financial claims. Native-limit pressure is explicit `UNKNOWN` unless a
future venue-native limit input is added.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_queue_churn"
TERMINAL_ACTIONS = {"cancel", "replace", "cancel_all", "expire", "expired", "reject", "rejected"}
CHURN_ACTIONS = {"cancel", "replace", "cancel_all"}


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


def _canonical_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"bid", "buy"}:
        return "BID"
    if side in {"ask", "sell"}:
        return "ASK"
    return "UNKNOWN"


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _identity_keys(label: dict[str, Any]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if label.get("order_id_hash"):
        keys.append(("order_id_hash", str(label["order_id_hash"])))
    if label.get("client_order_id_hash"):
        keys.append(("client_order_id_hash", str(label["client_order_id_hash"])))
    if label.get("order_label_seq") is not None:
        keys.append(("label_seq", str(label["order_label_seq"])))
    source_tuple = (
        label.get("order_source_line"),
        label.get("order_source_t"),
        label.get("order_source_order_index"),
    )
    if all(value is not None for value in source_tuple):
        keys.append(("source", _stable_hash(source_tuple)))
    return keys


def _event_keys(label: dict[str, Any]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if label.get("order_id_hash"):
        keys.append(("order_id_hash", str(label["order_id_hash"])))
    if label.get("client_order_id_hash"):
        keys.append(("client_order_id_hash", str(label["client_order_id_hash"])))
    if label.get("label_seq") is not None:
        keys.append(("label_seq", str(label["label_seq"])))
    source_tuple = (label.get("source_line"), label.get("source_t"), label.get("source_order_index"))
    if all(value is not None for value in source_tuple):
        keys.append(("source", _stable_hash(source_tuple)))
    return keys


def _load_lifecycle_events(label_lake_run: Path) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]], int]:
    summary = _load_json(label_lake_run / "label_lake_summary.json")
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    lifecycle_count = 0
    for _, label in _iter_jsonl(label_lake_run / "labels.jsonl"):
        if label.get("label_type") != "ORDER_LIFECYCLE_LABEL":
            continue
        lifecycle_count += 1
        for key in _event_keys(label):
            index.setdefault(key, []).append(label)
    return summary, index, lifecycle_count


def _load_pfill_labels(pfill_outcome_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_json(pfill_outcome_run / "pfill_outcome_summary.json")
    if summary.get("approved_for_live") is True:
        raise ValueError(f"{pfill_outcome_run} is not an admissible non-live P_fill outcome input")
    labels = [
        label
        for _, label in _iter_jsonl(pfill_outcome_run / "pfill_order_labels.jsonl")
        if label.get("label_type") == "ORDER_PFILL_OUTCOME_LABEL"
    ]
    return summary, labels


def _match_events(
    pfill_label: dict[str, Any],
    lifecycle_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    matched: dict[int, dict[str, Any]] = {}
    matched_by: list[str] = []
    for key in _identity_keys(pfill_label):
        events = lifecycle_index.get(key, [])
        if not events:
            continue
        matched_by.append(key[0])
        for event in events:
            seq = event.get("label_seq")
            if seq is not None:
                matched[int(seq)] = event
    return list(matched.values()), matched_by


def _summarize_events(events: list[dict[str, Any]], pfill_label: dict[str, Any]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    source_ticks = [
        source_t
        for source_t in (_safe_int(event.get("source_t")) for event in events)
        if source_t is not None
    ]
    for event in events:
        action = str(event.get("action") or "UNKNOWN").lower()
        action_counts[action] = action_counts.get(action, 0) + 1
    churn_count = sum(action_counts.get(action, 0) for action in CHURN_ACTIONS)
    terminal_count = sum(action_counts.get(action, 0) for action in TERMINAL_ACTIONS)
    first_tick = min(source_ticks) if source_ticks else None
    last_tick = max(source_ticks) if source_ticks else None
    observed_ticks = last_tick - first_tick if first_tick is not None and last_tick is not None else None
    return {
        "lifecycle_event_count": len(events),
        "lifecycle_action_counts": action_counts,
        "place_event_count": action_counts.get("place", 0),
        "replace_event_count": action_counts.get("replace", 0),
        "cancel_event_count": action_counts.get("cancel", 0),
        "cancel_all_event_count": action_counts.get("cancel_all", 0),
        "terminal_event_count_from_lifecycle": terminal_count,
        "churn_event_count": churn_count,
        "queue_reset_proxy_event_count": action_counts.get("replace", 0),
        "first_lifecycle_source_t": first_tick,
        "last_lifecycle_source_t": last_tick,
        "observed_lifecycle_source_ticks": observed_ticks,
        "churn_events_per_observed_tick": (
            churn_count / observed_ticks if observed_ticks and observed_ticks > 0 else None
        ),
        "terminal_action_first_from_pfill": pfill_label.get("terminal_action_first"),
        "terminal_event_count_from_pfill": pfill_label.get("terminal_event_count"),
        "terminal_horizon_source_ticks_from_pfill": pfill_label.get("observed_horizon_source_ticks"),
    }


def _base_label(run_id: str, seq: int, timestamp_ns: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_seq": seq,
        "label_type": "QUEUE_CHURN_LABEL",
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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
    }


def _summary_gate_reason(counts: dict[str, int]) -> str:
    if counts["queue_churn_label_count"] == 0:
        return "queue_churn_missing_labels"
    if counts["native_limit_pressure_unknown_count"] > 0:
        return "queue_churn_native_limit_pressure_unknown"
    if counts["unmatched_lifecycle_count"] > 0:
        return "queue_churn_partial_lifecycle_join"
    return "queue_churn_requires_board_review"


def build_queue_churn_labels(
    *,
    label_lake_run: Path,
    pfill_outcome_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51C-QUEUE-CHURN-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    lake_summary, lifecycle_index, lifecycle_count = _load_lifecycle_events(label_lake_run)
    pfill_summary, pfill_labels = _load_pfill_labels(pfill_outcome_run)
    if lake_summary.get("source_telemetry_sha256") != pfill_summary.get("source_telemetry_sha256"):
        raise ValueError("label lake and P_fill outcome run must share source_telemetry_sha256")

    records: list[dict[str, Any]] = []
    counts = {
        "queue_churn_label_count": 0,
        "matched_lifecycle_count": 0,
        "unmatched_lifecycle_count": 0,
        "filled_order_count": 0,
        "terminal_not_filled_order_count": 0,
        "censored_order_count": 0,
        "orders_with_replace_count": 0,
        "orders_with_cancel_count": 0,
        "orders_with_churn_count": 0,
        "orders_with_terminal_horizon_count": 0,
        "native_limit_pressure_unknown_count": 0,
    }
    for seq, pfill_label in enumerate(pfill_labels, start=1):
        events, matched_by = _match_events(pfill_label, lifecycle_index)
        summary = _summarize_events(events, pfill_label)
        outcome_status = str(pfill_label.get("outcome_status") or "UNKNOWN")
        counts["queue_churn_label_count"] += 1
        if events:
            counts["matched_lifecycle_count"] += 1
        else:
            counts["unmatched_lifecycle_count"] += 1
        if outcome_status == "OBSERVED_FILLED":
            counts["filled_order_count"] += 1
        elif outcome_status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
            counts["terminal_not_filled_order_count"] += 1
        elif outcome_status == "CENSORED_OR_UNOBSERVED":
            counts["censored_order_count"] += 1
        if summary["replace_event_count"] > 0:
            counts["orders_with_replace_count"] += 1
        if summary["cancel_event_count"] > 0:
            counts["orders_with_cancel_count"] += 1
        if summary["churn_event_count"] > 0:
            counts["orders_with_churn_count"] += 1
        if summary["terminal_horizon_source_ticks_from_pfill"] is not None:
            counts["orders_with_terminal_horizon_count"] += 1
        counts["native_limit_pressure_unknown_count"] += 1
        record = _base_label(run_id, seq, timestamp_ns)
        record.update({
            "source": "phase51c_label_lake_and_pfill_outcome",
            "source_telemetry_sha256": lake_summary.get("source_telemetry_sha256"),
            "label_lake_run": str(label_lake_run),
            "pfill_outcome_run": str(pfill_outcome_run),
            "order_key": pfill_label.get("order_key"),
            "order_holdout_split": pfill_label.get("order_holdout_split"),
            "order_label_seq": pfill_label.get("order_label_seq"),
            "order_source_line": pfill_label.get("order_source_line"),
            "order_source_t": pfill_label.get("order_source_t"),
            "venue_id": pfill_label.get("venue_id"),
            "side": _canonical_side(pfill_label.get("side")),
            "price": pfill_label.get("price"),
            "size": pfill_label.get("size"),
            "outcome_status": outcome_status,
            "p_fill_outcome": pfill_label.get("p_fill_outcome"),
            "fill_count": pfill_label.get("fill_count"),
            "filled_size_total": pfill_label.get("filled_size_total"),
            "lifecycle_join_status": "JOINED" if events else "MISSING",
            "lifecycle_join_key_types": sorted(set(matched_by)),
            "native_limit_pressure_status": "UNKNOWN_NO_NATIVE_LIMIT_PRESSURE_INPUT",
            "native_limit_pressure_hold_reason": "requires_phase51b_native_limit_pressure_join",
            "queue_churn_hold_reason": "requires_queue_reset_and_native_limit_pressure_calibration",
            **summary,
        })
        records.append(record)

    labels_path = out_dir / "queue_churn_labels.jsonl"
    summary_path = out_dir / "queue_churn_summary.json"
    _write_jsonl(labels_path, records)
    gate_reason = _summary_gate_reason(counts)
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
        "label_lake_run": str(label_lake_run),
        "pfill_outcome_run": str(pfill_outcome_run),
        "source_telemetry_sha256": lake_summary.get("source_telemetry_sha256"),
        "label_lake_summary_sha256": _sha256_file(label_lake_run / "label_lake_summary.json"),
        "pfill_outcome_summary_sha256": _sha256_file(pfill_outcome_run / "pfill_outcome_summary.json"),
        "order_lifecycle_labels_in_lake": lifecycle_count,
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
    parser.add_argument("--pfill-outcome-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_queue_churn_labels(
            label_lake_run=args.label_lake_run,
            pfill_outcome_run=args.pfill_outcome_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51c_queue_churn_labels: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_queue_churn_labels: wrote {out_dir}")
    print("phase51c_queue_churn_labels: status HOLD (queue/churn proxy evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
