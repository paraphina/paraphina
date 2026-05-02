#!/usr/bin/env python3
"""Build a Phase 5.1c deterministic join and holdout evidence pack.

This tool joins source-aligned quote, order, fill, markout, and balance labels.
It is an offline evidence gate only: no live orders, no model-training approval,
and no financial authority.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_join_holdout"


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


def _side_to_quote_side(side: Any) -> str:
    side_s = str(side or "").strip().lower()
    if side_s in {"bid", "buy"}:
        return "BID"
    if side_s in {"ask", "sell"}:
        return "ASK"
    return "UNKNOWN"


def _split_for_fill(fill_id: str) -> str:
    return "HOLDOUT" if int(hashlib.sha256(fill_id.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 else "TRAIN"


def _base_record(run_id: str, seq: int, timestamp_ns: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "join_seq": seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "timestamp_local_ns": timestamp_ns + seq,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_model_training": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "admissible_for_financial_claim": False,
    }


def _load_label_lake(label_lake_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    summary = _load_json(label_lake_run / "label_lake_summary.json")
    quote_by_key: dict[str, dict[str, Any]] = {}
    orders_by_decision: dict[str, dict[str, Any]] = {}
    orders_by_order_hash: dict[str, dict[str, Any]] = {}
    orders_by_client_hash: dict[str, dict[str, Any]] = {}
    for _, label in _iter_jsonl(label_lake_run / "labels.jsonl"):
        if label.get("label_type") == "QUOTE_DECISION_LABEL":
            key = _stable_hash([
                label.get("source_line"),
                label.get("source_t"),
                label.get("venue_id"),
                _side_to_quote_side(label.get("side")),
            ])
            quote_by_key.setdefault(key, label)
        elif label.get("label_type") == "ORDER_LIFECYCLE_LABEL":
            decision_id = label.get("decision_id")
            if decision_id:
                orders_by_decision.setdefault(str(decision_id), label)
            order_hash = label.get("order_id_hash")
            if order_hash:
                orders_by_order_hash.setdefault(str(order_hash), label)
            client_hash = label.get("client_order_id_hash")
            if client_hash:
                orders_by_client_hash.setdefault(str(client_hash), label)
    return summary, quote_by_key, orders_by_decision, {**orders_by_order_hash, **orders_by_client_hash}


def _load_observed(observed_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int], int]:
    summary = _load_json(observed_run / "observed_label_summary.json")
    fills: list[dict[str, Any]] = []
    markouts_by_fill: dict[str, int] = {}
    balance_labels = 0
    for _, label in _iter_jsonl(observed_run / "labels.jsonl"):
        label_type = label.get("label_type")
        if label_type == "OBSERVED_FILL_LABEL":
            fills.append(label)
        elif label_type == "OBSERVED_MARKOUT_LABEL":
            fill_id = label.get("fill_id")
            if fill_id:
                markouts_by_fill[str(fill_id)] = markouts_by_fill.get(str(fill_id), 0) + 1
        elif label_type == "BALANCE_RECONCILIATION_LABEL":
            balance_labels += 1
    return summary, fills, markouts_by_fill, balance_labels


def _join_order(
    fill: dict[str, Any],
    orders_by_decision: dict[str, dict[str, Any]],
    orders_by_hash: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    decision_id = fill.get("decision_id")
    if decision_id and str(decision_id) in orders_by_decision:
        return orders_by_decision[str(decision_id)], "decision_id"
    for key in ("order_id_hash", "client_order_id_hash"):
        value = fill.get(key)
        if value and str(value) in orders_by_hash:
            return orders_by_hash[str(value)], key
    return None, "unmatched"


def _join_candidate(fill: dict[str, Any], order: dict[str, Any] | None, quote_by_key: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    if not order:
        return None, "missing_order_join"
    key = _stable_hash([
        order.get("source_line"),
        order.get("source_t"),
        order.get("venue_id"),
        _side_to_quote_side(order.get("side")),
    ])
    candidate = quote_by_key.get(key)
    if candidate:
        return candidate, "source_line_source_t_venue_side"
    return None, "unmatched"


def _gate_reason(
    *,
    fill_count: int,
    complete_join_count: int,
    markout_join_count: int,
    balance_labels: int,
    maker_taker_unknown: int,
    holdout_count: int,
) -> str:
    if fill_count == 0:
        return "deterministic_join_missing_fills"
    if complete_join_count == 0:
        return "deterministic_join_missing_quote_order_fill_join"
    if markout_join_count < fill_count:
        return "deterministic_join_partial_markout_join"
    if balance_labels == 0:
        return "deterministic_join_missing_balance_reconciliation"
    if maker_taker_unknown > 0:
        return "deterministic_join_partial_maker_taker_attribution"
    if holdout_count == 0:
        return "deterministic_join_missing_holdout"
    return "deterministic_join_requires_board_review"


def build_join_holdout(
    *,
    label_lake_run: Path,
    observed_run: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51C-DETERMINISTIC-JOIN-HOLDOUT-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    lake_summary, quote_by_key, orders_by_decision, orders_by_hash = _load_label_lake(label_lake_run)
    observed_summary, fills, markouts_by_fill, balance_labels = _load_observed(observed_run)
    lake_source_sha = lake_summary.get("source_telemetry_sha256")
    observed_source_sha = observed_summary.get("source_telemetry_sha256")
    if lake_source_sha != observed_source_sha:
        raise ValueError("label lake and observed labels must share source_telemetry_sha256")

    joined: list[dict[str, Any]] = []
    counts = {
        "fill_labels": len(fills),
        "order_join_count": 0,
        "candidate_join_count": 0,
        "complete_join_count": 0,
        "markout_join_count": 0,
        "maker_taker_unknown_count": 0,
        "train_count": 0,
        "holdout_count": 0,
    }
    reason_counts: dict[str, int] = {}
    for seq, fill in enumerate(fills, start=1):
        order, order_join_key = _join_order(fill, orders_by_decision, orders_by_hash)
        candidate, candidate_join_key = _join_candidate(fill, order, quote_by_key)
        split = _split_for_fill(str(fill.get("fill_id")))
        markout_count = markouts_by_fill.get(str(fill.get("fill_id")), 0)
        maker_taker_unknown = fill.get("maker_taker_role") == "UNKNOWN"
        if order:
            counts["order_join_count"] += 1
        if candidate:
            counts["candidate_join_count"] += 1
        if order and candidate and markout_count > 0:
            counts["complete_join_count"] += 1
        if markout_count > 0:
            counts["markout_join_count"] += 1
        if maker_taker_unknown:
            counts["maker_taker_unknown_count"] += 1
        counts["holdout_count" if split == "HOLDOUT" else "train_count"] += 1
        unmatched_reasons: list[str] = []
        if not order:
            unmatched_reasons.append("missing_order_join")
        if not candidate:
            unmatched_reasons.append("missing_candidate_join")
        if markout_count == 0:
            unmatched_reasons.append("missing_markout_join")
        if maker_taker_unknown:
            unmatched_reasons.append("maker_taker_unknown")
        if not unmatched_reasons:
            unmatched_reasons.append("complete_join")
        for reason in unmatched_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        record = _base_record(run_id, seq, timestamp_ns)
        record.update({
            "label_type": "DETERMINISTIC_JOIN_LABEL",
            "fill_id": fill.get("fill_id"),
            "venue_id": fill.get("venue_id"),
            "side": fill.get("side"),
            "price": fill.get("price"),
            "size": fill.get("size"),
            "fill_time_ms": fill.get("fill_time_ms"),
            "maker_taker_role": fill.get("maker_taker_role"),
            "maker_taker_attribution_status": fill.get("maker_taker_attribution_status"),
            "maker_taker_attribution_source": fill.get("maker_taker_attribution_source"),
            "order_join_status": "JOINED" if order else "MISSING",
            "order_join_key": order_join_key,
            "candidate_join_status": "JOINED" if candidate else "MISSING",
            "candidate_join_key": candidate_join_key,
            "candidate_id": candidate.get("candidate_id") if candidate else None,
            "order_decision_id": order.get("decision_id") if order else None,
            "fill_decision_id": fill.get("decision_id"),
            "markout_join_count": markout_count,
            "balance_reconciliation_available": balance_labels > 0,
            "holdout_split": split,
            "join_status": "COMPLETE_FOR_NONLIVE_REVIEW" if order and candidate and markout_count > 0 else "PARTIAL",
            "unmatched_reason_list": unmatched_reasons,
            "admissible_for_model_training": False,
            "training_hold_reason": "requires_board_review_and_role_attribution_resolution",
        })
        joined.append(record)

    gate_reason = _gate_reason(
        fill_count=counts["fill_labels"],
        complete_join_count=counts["complete_join_count"],
        markout_join_count=counts["markout_join_count"],
        balance_labels=balance_labels,
        maker_taker_unknown=counts["maker_taker_unknown_count"],
        holdout_count=counts["holdout_count"],
    )
    joined_path = out_dir / "joined_labels.jsonl"
    summary_path = out_dir / "join_holdout_summary.json"
    _write_jsonl(joined_path, joined)
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
        "observed_run": str(observed_run),
        "source_telemetry_sha256": lake_source_sha,
        "ev_shadow_telemetry_sha256": lake_summary.get("ev_shadow_telemetry_sha256"),
        "quote_decision_labels": lake_summary.get("quote_decision_labels", 0),
        "order_lifecycle_labels": lake_summary.get("order_lifecycle_labels", 0),
        "markout_labels": observed_summary.get("markout_labels", 0),
        "balance_reconciliation_labels": balance_labels,
        "maker_taker_role_counts": observed_summary.get("maker_taker_role_counts", {}),
        "reason_counts": reason_counts,
        **counts,
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [joined_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [joined_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-lake-run", type=Path, required=True)
    parser.add_argument("--observed-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_join_holdout(
            label_lake_run=args.label_lake_run,
            observed_run=args.observed_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51c_join_holdout: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_join_holdout: wrote {out_dir}")
    print("phase51c_join_holdout: status HOLD (deterministic join evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
