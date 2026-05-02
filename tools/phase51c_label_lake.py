#!/usr/bin/env python3
"""Build a Phase 5.1c non-live calibration label lake scaffold.

The output is an immutable, provenance-linked label dataset for review. It is
not a training artifact, not a profitability claim, and not live authorization.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_label_lake"


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


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, record


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


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _base_label(
    *,
    run_id: str,
    label_seq: int,
    label_type: str,
    timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_seq": label_seq,
        "label_type": label_type,
        "timestamp_local_ns": timestamp_ns + label_seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "admissible_for_financial_claim": False,
        "admissible_for_model_training": False,
    }


def _record_safe_nonlive(record: dict[str, Any]) -> bool:
    return (
        record.get("no_live_flag") is True
        and record.get("approved_for_live") is False
        and record.get("approved_for_canary") is False
        and record.get("approved_for_capital_escalation") is False
        and record.get("live_orders_allowed") is False
        and record.get("capital_change_allowed") is False
        and record.get("risk_limit_relaxation_allowed") is False
        and record.get("admissible_for_financial_claim") in (None, False)
    )


def _quote_decision_labels(
    *,
    ev_shadow_telemetry: Path,
    run_id: str,
    timestamp_ns: int,
    start_seq: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels: list[dict[str, Any]] = []
    decisions: dict[str, int] = {}
    for line_no, record in _iter_jsonl(ev_shadow_telemetry):
        if record.get("event_type") != "V2_EV_EVALUATED":
            continue
        if not _record_safe_nonlive(record):
            raise ValueError(f"unsafe EV shadow record flags at {ev_shadow_telemetry}:{line_no}")
        decision = str(record.get("decision") or "UNKNOWN")
        decisions[decision] = decisions.get(decision, 0) + 1
        label = _base_label(
            run_id=run_id,
            label_seq=start_seq + len(labels),
            label_type="QUOTE_DECISION_LABEL",
            timestamp_ns=timestamp_ns,
        )
        label.update({
            "source": "phase51_ev_shadow",
            "source_line": record.get("source_line", line_no),
            "source_t": record.get("source_t"),
            "source_record_sha256": record.get("source_record_sha256"),
            "candidate_id": record.get("candidate_id"),
            "venue_id": record.get("venue_id"),
            "side": record.get("side"),
            "layer": record.get("layer"),
            "decision": decision,
            "decision_reason_primary": record.get("decision_reason_primary"),
            "decision_reason_secondary_list": record.get("decision_reason_secondary_list", []),
            "calibration_bucket_id": record.get("calibration_bucket_id"),
            "calibration_status": record.get("calibration_status"),
            "label_status": "COUNTERFACTUAL_DECISION_ONLY",
            "label_confidence": 0.0,
            "training_hold_reason": "requires_observed_fill_markout_and_holdout_calibration",
        })
        labels.append(label)
    return labels, decisions


def _order_lifecycle_labels(
    *,
    source_telemetry: Path,
    run_id: str,
    timestamp_ns: int,
    start_seq: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels: list[dict[str, Any]] = []
    actions: dict[str, int] = {}
    for line_no, record in _iter_jsonl(source_telemetry):
        source_t = record.get("t")
        for order_index, order in enumerate(record.get("orders") or []):
            if not isinstance(order, dict):
                continue
            action = str(order.get("action") or "UNKNOWN")
            actions[action] = actions.get(action, 0) + 1
            label = _base_label(
                run_id=run_id,
                label_seq=start_seq + len(labels),
                label_type="ORDER_LIFECYCLE_LABEL",
                timestamp_ns=timestamp_ns,
            )
            label.update({
                "source": "phase5_telemetry_orders",
                "source_line": line_no,
                "source_t": source_t,
                "source_record_sha256": _stable_hash(order),
                "source_order_index": order_index,
                "venue_id": order.get("venue_id"),
                "action": action,
                "status": order.get("status"),
                "decision_id": order.get("decision_id"),
                "order_id_hash": _stable_hash(order.get("order_id")) if order.get("order_id") else None,
                "client_order_id_hash": _stable_hash(order.get("client_order_id")) if order.get("client_order_id") else None,
                "side": order.get("side"),
                "price": order.get("price"),
                "size": order.get("size"),
                "post_only": order.get("post_only"),
                "reduce_only": order.get("reduce_only"),
                "label_status": "OBSERVED_ORDER_LIFECYCLE",
                "label_confidence": 1.0,
                "training_hold_reason": "order_lifecycle_label_requires_fill_markout_join_before_training",
            })
            labels.append(label)
    return labels, actions


def _coverage_label(
    *,
    run_id: str,
    timestamp_ns: int,
    label_seq: int,
    counts: dict[str, int],
    decision_counts: dict[str, int],
    action_counts: dict[str, int],
    phase51b_acceptance: dict[str, Any],
) -> dict[str, Any]:
    label = _base_label(
        run_id=run_id,
        label_seq=label_seq,
        label_type="LABEL_COVERAGE_SUMMARY",
        timestamp_ns=timestamp_ns,
    )
    native_limit_unknown = "lighter_open_order_limit_headroom_unknown" in phase51b_acceptance.get("limitations", [])
    label.update({
        "label_status": "HOLD_FOR_COVERAGE",
        "quote_decision_labels": counts.get("quote_decision_labels", 0),
        "order_lifecycle_labels": counts.get("order_lifecycle_labels", 0),
        "fill_labels": counts.get("fill_labels", 0),
        "markout_labels": counts.get("markout_labels", 0),
        "balance_reconciliation_labels": counts.get("balance_reconciliation_labels", 0),
        "quote_decision_counts": decision_counts,
        "order_action_counts": action_counts,
        "fill_label_status": "MISSING",
        "markout_label_status": "MISSING",
        "balance_reconciliation_status": "MISSING",
        "native_limit_pressure_status": "UNKNOWN" if native_limit_unknown else "OBSERVED",
        "phase51b_acceptance_status": phase51b_acceptance.get("status"),
        "approved_for_calibration_label_ingestion": phase51b_acceptance.get(
            "approved_for_calibration_label_ingestion"
        ) is True,
        "training_hold_reason": "missing_fill_markout_balance_labels",
    })
    return label


def build_label_lake(
    *,
    source_telemetry: Path,
    ev_shadow_telemetry: Path,
    phase51b_acceptance_path: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    phase51b_acceptance = _load_json(phase51b_acceptance_path)
    if phase51b_acceptance.get("approved_for_calibration_label_ingestion") is not True:
        raise ValueError("Phase 5.1b acceptance must approve calibration-label ingestion")
    if (
        phase51b_acceptance.get("approved_for_live") is not False
        or phase51b_acceptance.get("approved_for_canary") is not False
        or phase51b_acceptance.get("approved_for_capital_escalation") is not False
        or phase51b_acceptance.get("approved_for_financial_claim") is not False
    ):
        raise ValueError("Phase 5.1b acceptance must keep live/canary/capital/financial claims blocked")
    run_id = run_id or f"PHASE51C-LABEL-LAKE-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    labels: list[dict[str, Any]] = []
    quote_labels, decision_counts = _quote_decision_labels(
        ev_shadow_telemetry=ev_shadow_telemetry,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        start_seq=1,
    )
    labels.extend(quote_labels)
    order_labels, action_counts = _order_lifecycle_labels(
        source_telemetry=source_telemetry,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        start_seq=len(labels) + 1,
    )
    labels.extend(order_labels)
    counts = {
        "quote_decision_labels": len(quote_labels),
        "order_lifecycle_labels": len(order_labels),
        "fill_labels": 0,
        "markout_labels": 0,
        "balance_reconciliation_labels": 0,
    }
    labels.append(_coverage_label(
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        label_seq=len(labels) + 1,
        counts=counts,
        decision_counts=decision_counts,
        action_counts=action_counts,
        phase51b_acceptance=phase51b_acceptance,
    ))

    labels_path = out_dir / "labels.jsonl"
    summary_path = out_dir / "label_lake_summary.json"
    manifest_path = out_dir / "manifest.json"
    _write_jsonl(labels_path, labels)
    summary = {
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "label_lake_scaffold_missing_fill_markout_balance_coverage",
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "source_telemetry": str(source_telemetry),
        "source_telemetry_sha256": _sha256_file(source_telemetry),
        "ev_shadow_telemetry": str(ev_shadow_telemetry),
        "ev_shadow_telemetry_sha256": _sha256_file(ev_shadow_telemetry),
        "phase51b_acceptance": str(phase51b_acceptance_path),
        "phase51b_acceptance_sha256": _sha256_file(phase51b_acceptance_path),
        **counts,
        "quote_decision_counts": decision_counts,
        "order_action_counts": action_counts,
        "fill_label_status": "MISSING",
        "markout_label_status": "MISSING",
        "balance_reconciliation_status": "MISSING",
        "native_limit_pressure_status": (
            "UNKNOWN"
            if "lighter_open_order_limit_headroom_unknown" in phase51b_acceptance.get("limitations", [])
            else "OBSERVED"
        ),
        "record_count": len(labels),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, summary_path]),
    })
    manifest = {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, summary_path, artifact_index_path]),
    }
    _write_json(manifest_path, manifest)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-telemetry", type=Path, required=True)
    parser.add_argument("--ev-shadow-telemetry", type=Path, required=True)
    parser.add_argument("--phase51b-acceptance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_label_lake(
            source_telemetry=args.source_telemetry,
            ev_shadow_telemetry=args.ev_shadow_telemetry,
            phase51b_acceptance_path=args.phase51b_acceptance,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51c_label_lake: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_label_lake: wrote {out_dir}")
    print("phase51c_label_lake: status HOLD (label scaffold only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
