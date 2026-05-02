#!/usr/bin/env python3
"""Audit remaining Lighter maker/taker attribution gaps for Phase 5.1c.

This offline evidence gate compares observed Lighter fills against sanitized
native Lighter trade backfill. It explains why fills remain UNKNOWN without
inferring maker/taker role from quote intent and without approving live,
canary, model training, EV admission, or financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_lighter_attribution_gap_audit"
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
GAP_REASONS = {
    "ATTRIBUTED_NATIVE_ROLE",
    "NO_NATIVE_TRADE_MATCH",
    "ORDER_ID_MISMATCH",
    "CLIENT_ID_MISMATCH",
    "TIME_PRICE_SIZE_AMBIGUOUS",
    "OUTSIDE_BACKFILL_WINDOW",
    "PARSER_GAP",
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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_ids(*values: Any) -> list[str]:
    ids: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            ids.append(text)
    return ids


def _extract_items(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _trade_timestamp_ms(trade: dict[str, Any]) -> int | None:
    return _as_int(trade.get("timestamp") or trade.get("time") or trade.get("created_at"))


def _trade_role_for_account(trade: dict[str, Any], account_index: int | None) -> str | None:
    is_maker_ask = trade.get("is_maker_ask")
    if is_maker_ask is None:
        is_maker_ask = trade.get("isMakerAsk")
    if not isinstance(is_maker_ask, bool) or account_index is None:
        return None
    ask_account_id = _as_int(trade.get("ask_account_id") or trade.get("askAccountId"))
    bid_account_id = _as_int(trade.get("bid_account_id") or trade.get("bidAccountId"))
    if account_index == ask_account_id:
        return "MAKER" if is_maker_ask else "TAKER"
    if account_index == bid_account_id:
        return "TAKER" if is_maker_ask else "MAKER"
    return None


def _trade_order_ids(trade: dict[str, Any]) -> list[str]:
    return _string_ids(
        trade.get("ask_id"),
        trade.get("ask_id_str"),
        trade.get("ask_client_id"),
        trade.get("ask_client_id_str"),
        trade.get("bid_id"),
        trade.get("bid_id_str"),
        trade.get("bid_client_id"),
        trade.get("bid_client_id_str"),
    )


def _trade_side_ids(trade: dict[str, Any], side: str) -> list[str]:
    if side == "ASK":
        return _string_ids(trade.get("ask_id"), trade.get("ask_id_str"), trade.get("ask_client_id"), trade.get("ask_client_id_str"))
    if side == "BID":
        return _string_ids(trade.get("bid_id"), trade.get("bid_id_str"), trade.get("bid_client_id"), trade.get("bid_client_id_str"))
    return []


def _canonical_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"ask", "sell"}:
        return "ASK"
    if side in {"bid", "buy"}:
        return "BID"
    return "UNKNOWN"


def _load_observed_fills(observed_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = observed_run / "observed_label_summary.json"
    labels_path = observed_run / "labels.jsonl"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    fills: list[dict[str, Any]] = []
    for _, record in _iter_jsonl(labels_path):
        if record.get("label_type") != "OBSERVED_FILL_LABEL":
            continue
        _check_unsafe(record, labels_path, label="label")
        if str(record.get("venue_id") or "").lower() == "lighter":
            fills.append(record)
    return summary, fills


def _load_join_by_fill(join_run: Path, expected_source_sha: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary_path = join_run / "join_holdout_summary.json"
    labels_path = join_run / "joined_labels.jsonl"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    if summary.get("source_telemetry_sha256") != expected_source_sha:
        raise ValueError(f"{summary_path} source_telemetry_sha256 mismatch")
    _check_unsafe(summary, summary_path, label="summary")
    by_fill: dict[str, dict[str, Any]] = {}
    for _, record in _iter_jsonl(labels_path):
        if record.get("label_type") != "DETERMINISTIC_JOIN_LABEL":
            continue
        _check_unsafe(record, labels_path, label="label")
        fill_id = record.get("fill_id")
        if fill_id:
            by_fill[str(fill_id)] = record
    return summary, by_fill


def _load_trade_backfill(trade_backfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = trade_backfill_run / "lighter_trade_backfill_summary.json"
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    trades_path = Path(str(summary.get("trades_path") or ""))
    if not trades_path.is_absolute():
        trades_path = ROOT / trades_path
    payload = _load_json(trades_path)
    trades = _extract_items(payload, {"trades", "trade_history", "tradehistory"})
    return summary, trades


def _identity_match(fill: dict[str, Any], trades_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for order_id in _string_ids(fill.get("order_id_hash"), fill.get("client_order_id_hash")):
        trade = trades_by_id.get(order_id)
        if trade is not None:
            return trade
    return None


def _time_price_size_candidates(
    fill: dict[str, Any],
    trades: list[dict[str, Any]],
    *,
    time_tolerance_ms: int,
    price_tolerance: float,
    size_tolerance: float,
) -> list[dict[str, Any]]:
    fill_ts = _as_int(fill.get("fill_time_ms"))
    fill_price = _as_float(fill.get("price"))
    fill_size = _as_float(fill.get("size"))
    if fill_ts is None or fill_price is None or fill_size is None:
        return []
    candidates: list[dict[str, Any]] = []
    for trade in trades:
        ts = _trade_timestamp_ms(trade)
        price = _as_float(trade.get("price"))
        size = _as_float(trade.get("size"))
        if ts is None or price is None or size is None:
            continue
        if abs(ts - fill_ts) <= time_tolerance_ms and abs(price - fill_price) <= price_tolerance and abs(size - fill_size) <= size_tolerance:
            candidates.append(trade)
    return candidates


def _classify_gap(
    fill: dict[str, Any],
    *,
    trades: list[dict[str, Any]],
    trades_by_id: dict[str, dict[str, Any]],
    account_index: int | None,
    trade_min_ts: int | None,
    trade_max_ts: int | None,
    time_tolerance_ms: int,
    price_tolerance: float,
    size_tolerance: float,
) -> tuple[str, str, str | None, dict[str, Any]]:
    existing_role = str(fill.get("maker_taker_role") or "UNKNOWN")
    if existing_role in {"MAKER", "TAKER"}:
        return "ATTRIBUTED_NATIVE_ROLE", "role already attributed in observed label", existing_role, {}
    fill_ts = _as_int(fill.get("fill_time_ms"))
    side = _canonical_side(fill.get("side"))
    identity_trade = _identity_match(fill, trades_by_id)
    if identity_trade is not None:
        role = _trade_role_for_account(identity_trade, account_index)
        if role:
            return "ATTRIBUTED_NATIVE_ROLE", "native identity match provides role but observed label is stale", role, {
                "matched_trade_id": identity_trade.get("trade_id") or identity_trade.get("trade_id_str"),
            }
        return "PARSER_GAP", "identity match exists but account role is not derivable", None, {}
    if fill_ts is None or trade_min_ts is None or trade_max_ts is None:
        return "PARSER_GAP", "missing fill or backfill timestamp bounds", None, {}
    if fill_ts < trade_min_ts - time_tolerance_ms or fill_ts > trade_max_ts + time_tolerance_ms:
        return "OUTSIDE_BACKFILL_WINDOW", "fill timestamp outside native trade backfill window", None, {}
    candidates = _time_price_size_candidates(
        fill,
        trades,
        time_tolerance_ms=time_tolerance_ms,
        price_tolerance=price_tolerance,
        size_tolerance=size_tolerance,
    )
    if len(candidates) > 1:
        return "TIME_PRICE_SIZE_AMBIGUOUS", "multiple native trades match time/price/size", None, {"candidate_count": len(candidates)}
    if len(candidates) == 1:
        candidate = candidates[0]
        side_ids = _trade_side_ids(candidate, side)
        all_ids = _trade_order_ids(candidate)
        if side_ids and not any(order_id in _string_ids(fill.get("order_id_hash"), fill.get("client_order_id_hash")) for order_id in side_ids):
            return "ORDER_ID_MISMATCH", "time/price/size candidate has different native side order id", None, {
                "matched_trade_id": candidate.get("trade_id") or candidate.get("trade_id_str"),
                "native_side_id_count": len(side_ids),
            }
        if all_ids:
            return "CLIENT_ID_MISMATCH", "time/price/size candidate exists but no fill identity matches native ids", None, {
                "matched_trade_id": candidate.get("trade_id") or candidate.get("trade_id_str"),
                "native_id_count": len(all_ids),
            }
        return "PARSER_GAP", "time/price/size candidate lacks usable native ids", None, {}
    return "NO_NATIVE_TRADE_MATCH", "no native trade matches identity or time/price/size", None, {}


def build_lighter_attribution_gap_audit(
    *,
    observed_run: Path,
    join_holdout_run: Path,
    lighter_trade_backfill_run: Path,
    phase51b_native_run: Path | None,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    time_tolerance_ms: int,
    price_tolerance: float,
    size_tolerance: float,
) -> Path:
    run_id = run_id or f"PHASE51C-LIGHTER-ATTRIBUTION-GAP-AUDIT-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    observed_summary, fills = _load_observed_fills(observed_run)
    join_summary, join_by_fill = _load_join_by_fill(join_holdout_run, str(observed_summary.get("source_telemetry_sha256")))
    trade_summary, trades = _load_trade_backfill(lighter_trade_backfill_run)
    phase51b_acceptance: dict[str, Any] | None = None
    phase51b_acceptance_sha256: str | None = None
    if phase51b_native_run is not None:
        phase51b_path = phase51b_native_run / "phase51b_acceptance.json"
        phase51b_acceptance = _load_json(phase51b_path)
        _check_unsafe(phase51b_acceptance, phase51b_path, label="summary")
        phase51b_acceptance_sha256 = _sha256_file(phase51b_path)

    account_index = _as_int(trade_summary.get("account_index"))
    trade_timestamps = [ts for ts in (_trade_timestamp_ms(trade) for trade in trades) if ts is not None]
    trade_min_ts = min(trade_timestamps) if trade_timestamps else None
    trade_max_ts = max(trade_timestamps) if trade_timestamps else None
    trades_by_id: dict[str, dict[str, Any]] = {}
    for trade in trades:
        for order_id in _trade_order_ids(trade):
            trades_by_id.setdefault(order_id, trade)

    records: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    stale_upgrade_count = 0
    for fill in sorted(fills, key=lambda row: str(row.get("fill_id") or "")):
        gap_reason, detail, native_role, extra = _classify_gap(
            fill,
            trades=trades,
            trades_by_id=trades_by_id,
            account_index=account_index,
            trade_min_ts=trade_min_ts,
            trade_max_ts=trade_max_ts,
            time_tolerance_ms=time_tolerance_ms,
            price_tolerance=price_tolerance,
            size_tolerance=size_tolerance,
        )
        if gap_reason not in GAP_REASONS:
            raise ValueError(f"unexpected gap_reason {gap_reason}")
        reason_counts[gap_reason] = reason_counts.get(gap_reason, 0) + 1
        observed_role = str(fill.get("maker_taker_role") or "UNKNOWN")
        role_counts[observed_role] = role_counts.get(observed_role, 0) + 1
        if observed_role == "UNKNOWN" and native_role in {"MAKER", "TAKER"}:
            stale_upgrade_count += 1
        join = join_by_fill.get(str(fill.get("fill_id") or ""), {})
        record = {
            "schema_version": 1,
            "label_type": "LIGHTER_ATTRIBUTION_GAP_AUDIT_LABEL",
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "timestamp_local_ns": timestamp_ns + len(records) + 1,
            "source_telemetry_sha256": observed_summary.get("source_telemetry_sha256"),
            "fill_id": fill.get("fill_id"),
            "fill_time_ms": fill.get("fill_time_ms"),
            "venue_id": "lighter",
            "side": _canonical_side(fill.get("side")),
            "price": fill.get("price"),
            "size": fill.get("size"),
            "observed_maker_taker_role": observed_role,
            "native_role_if_determinable": native_role,
            "gap_reason": gap_reason,
            "gap_reason_detail": detail,
            "candidate_join_status": join.get("candidate_join_status"),
            "join_status": join.get("join_status"),
            "order_id_hash_present": bool(fill.get("order_id_hash")),
            "client_order_id_hash_present": bool(fill.get("client_order_id_hash")),
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
            **extra,
        }
        records.append(record)

    gate_reason = (
        "lighter_attribution_gap_stale_unknowns_upgradable"
        if stale_upgrade_count > 0
        else "lighter_attribution_gap_unknowns_unresolved"
        if reason_counts.get("NO_NATIVE_TRADE_MATCH", 0) or reason_counts.get("OUTSIDE_BACKFILL_WINDOW", 0)
        else "lighter_attribution_gap_audit_complete_hold"
    )
    labels_path = out_dir / "lighter_attribution_gap_labels.jsonl"
    summary_path = out_dir / "lighter_attribution_gap_summary.json"
    _write_jsonl(labels_path, records)
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
        "observed_run": str(observed_run),
        "join_holdout_run": str(join_holdout_run),
        "lighter_trade_backfill_run": str(lighter_trade_backfill_run),
        "phase51b_native_run": str(phase51b_native_run) if phase51b_native_run else None,
        "phase51b_acceptance_sha256": phase51b_acceptance_sha256,
        "phase51b_approved_for_calibration_label_ingestion": (
            phase51b_acceptance.get("approved_for_calibration_label_ingestion") if phase51b_acceptance else None
        ),
        "source_telemetry_sha256": observed_summary.get("source_telemetry_sha256"),
        "observed_summary_sha256": _sha256_file(observed_run / "observed_label_summary.json"),
        "observed_labels_sha256": _sha256_file(observed_run / "labels.jsonl"),
        "join_holdout_summary_sha256": _sha256_file(join_holdout_run / "join_holdout_summary.json"),
        "joined_labels_sha256": _sha256_file(join_holdout_run / "joined_labels.jsonl"),
        "lighter_trade_backfill_summary_sha256": _sha256_file(lighter_trade_backfill_run / "lighter_trade_backfill_summary.json"),
        "lighter_trade_backfill_trades_sha256": trade_summary.get("trades_sha256"),
        "lighter_fill_count": len(records),
        "observed_role_counts": dict(sorted(role_counts.items())),
        "gap_reason_counts": dict(sorted(reason_counts.items())),
        "stale_unknowns_upgradable_from_native_identity": stale_upgrade_count,
        "native_trade_count": len(trades),
        "native_trade_timestamp_min_ms": trade_min_ts,
        "native_trade_timestamp_max_ms": trade_max_ts,
        "time_tolerance_ms": time_tolerance_ms,
        "price_tolerance": price_tolerance,
        "size_tolerance": size_tolerance,
        "lighter_attribution_gap_labels_sha256": _sha256_file(labels_path),
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
    parser.add_argument("--observed-run", type=Path, required=True)
    parser.add_argument("--join-holdout-run", type=Path, required=True)
    parser.add_argument("--lighter-trade-backfill-run", type=Path, required=True)
    parser.add_argument("--phase51b-native-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--time-tolerance-ms", type=int, default=250)
    parser.add_argument("--price-tolerance", type=float, default=0.000001)
    parser.add_argument("--size-tolerance", type=float, default=0.000001)
    args = parser.parse_args()
    try:
        out_dir = build_lighter_attribution_gap_audit(
            observed_run=args.observed_run,
            join_holdout_run=args.join_holdout_run,
            lighter_trade_backfill_run=args.lighter_trade_backfill_run,
            phase51b_native_run=args.phase51b_native_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            time_tolerance_ms=args.time_tolerance_ms,
            price_tolerance=args.price_tolerance,
            size_tolerance=args.size_tolerance,
        )
    except Exception as exc:
        print(f"phase51c_lighter_attribution_gap_audit: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_lighter_attribution_gap_audit: wrote {out_dir}")
    print("phase51c_lighter_attribution_gap_audit: status HOLD (diagnostic attribution audit only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
