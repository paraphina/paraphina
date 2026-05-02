#!/usr/bin/env python3
"""Build Phase 5.1c observed fill/balance labels from existing artifacts.

This is a read-only evidence extractor. It does not submit orders, does not
promote models, and does not authorize live/canary/capital escalation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51c_observed_labels"


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


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _iter_json_stream(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            source = line.strip()
            if not source:
                continue
            pos = 0
            while pos < len(source):
                try:
                    record, end = decoder.raw_decode(source, pos)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}:{pos + 1}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_no}")
                yield line_no, record
                pos = end
                while pos < len(source) and source[pos].isspace():
                    pos += 1


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


def _hash_or_none(value: Any) -> str | None:
    return _stable_hash(value) if value not in (None, "") else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_time_ms(record: dict[str, Any]) -> int | None:
    for key in ("timestamp_local_ms", "timestamp_ms", "exchange_timestamp_ms", "kf_last_update_ms"):
        value = _safe_int(record.get(key))
        if value is not None:
            return value
    treasury = record.get("treasury_guidance")
    if isinstance(treasury, dict):
        value = _safe_int(treasury.get("as_of_ms"))
        if value is not None:
            return value
    return None


def _reference_price(record: dict[str, Any], venue_index: Any) -> tuple[float | None, str]:
    fair_value = _safe_float(record.get("fair_value"))
    if fair_value is not None:
        return fair_value, "fair_value"
    venue_i = _safe_int(venue_index)
    venue_mid = record.get("venue_mid_usd")
    if venue_i is not None and isinstance(venue_mid, list) and 0 <= venue_i < len(venue_mid):
        mid = _safe_float(venue_mid[venue_i])
        if mid is not None:
            return mid, "venue_mid_usd"
    return None, "missing_reference_price"


def _signed_markout_usd(side: Any, fill_price: Any, size: Any, future_price: float) -> float | None:
    px = _safe_float(fill_price)
    qty = _safe_float(size)
    if px is None or qty is None:
        return None
    side_s = str(side or "").lower()
    if side_s.startswith("buy") or side_s == "bid":
        return (future_price - px) * abs(qty)
    if side_s.startswith("sell") or side_s == "ask":
        return (px - future_price) * abs(qty)
    return None


def _extract_items(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        value = lowered.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _string_ids(*values: Any) -> list[str]:
    ids: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            ids.append(text)
    return ids


def _load_lighter_trade_role_index(path: Path | None) -> tuple[dict[str, str], str | None]:
    if path is None:
        return {}, None
    payload = _load_json(path)
    trades = _extract_items(payload, {"trades", "trade_history", "tradehistory"})
    index: dict[str, str] = {}
    for trade in trades:
        is_maker_ask = trade.get("is_maker_ask")
        if is_maker_ask is None:
            is_maker_ask = trade.get("isMakerAsk")
        if not isinstance(is_maker_ask, bool):
            continue
        ask_role = "MAKER" if is_maker_ask else "TAKER"
        bid_role = "TAKER" if is_maker_ask else "MAKER"
        for order_id in _string_ids(
            trade.get("ask_id"),
            trade.get("ask_id_str"),
            trade.get("ask_client_id"),
            trade.get("ask_client_id_str"),
        ):
            index[order_id] = ask_role
        for order_id in _string_ids(
            trade.get("bid_id"),
            trade.get("bid_id_str"),
            trade.get("bid_client_id"),
            trade.get("bid_client_id_str"),
        ):
            index[order_id] = bid_role
    return index, _sha256_file(path)


def _maker_taker_role(fill: dict[str, Any], lighter_trade_roles: dict[str, str]) -> tuple[str, str]:
    for key in ("maker_or_taker", "maker_taker", "liquidity", "role"):
        raw = fill.get(key)
        if isinstance(raw, str) and raw.strip():
            lowered = raw.strip().lower()
            if "maker" in lowered:
                return "MAKER", "telemetry_fill_field"
            if "taker" in lowered:
                return "TAKER", "telemetry_fill_field"
    if fill.get("is_maker") is True:
        return "MAKER", "telemetry_fill_field"
    if fill.get("is_maker") is False:
        return "TAKER", "telemetry_fill_field"
    if str(fill.get("venue_id") or "").lower() == "lighter":
        for order_id in _string_ids(fill.get("order_id"), fill.get("client_order_id")):
            role = lighter_trade_roles.get(order_id)
            if role:
                return role, "lighter_trades_json"
    return "UNKNOWN", "unknown"


def _extract_fill_labels(
    *,
    source_telemetry: Path,
    run_id: str,
    timestamp_ns: int,
    markout_horizons_ms: list[int],
    lighter_trade_roles: dict[str, str],
    labels_file,
) -> tuple[int, int, dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
    fill_count = 0
    markout_count = 0
    label_seq = 0
    per_venue: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    role_counts_by_venue: dict[str, dict[str, int]] = {}
    pending_markouts: list[dict[str, Any]] = []
    for line_no, record in _iter_json_stream(source_telemetry):
        record_time_ms = _record_time_ms(record)
        if record_time_ms is not None and pending_markouts:
            still_pending: list[dict[str, Any]] = []
            for pending in pending_markouts:
                remaining_horizons: list[int] = []
                for horizon_ms in pending["remaining_horizons_ms"]:
                    if record_time_ms < pending["fill_time_ms"] + horizon_ms:
                        remaining_horizons.append(horizon_ms)
                        continue
                    future_price, price_source = _reference_price(record, pending["venue_index"])
                    markout = (
                        _signed_markout_usd(
                            pending["side"],
                            pending["price"],
                            pending["size"],
                            future_price,
                        )
                        if future_price is not None
                        else None
                    )
                    if markout is None:
                        remaining_horizons.append(horizon_ms)
                        continue
                    markout_count += 1
                    label_seq += 1
                    markout_label = _base_label(
                        run_id=run_id,
                        label_seq=label_seq,
                        label_type="OBSERVED_MARKOUT_LABEL",
                        timestamp_ns=timestamp_ns,
                    )
                    markout_label.update({
                        "source": "phase5_telemetry_future_reference",
                        "source_line": pending["source_line"],
                        "source_t": pending["source_t"],
                        "source_fill_index": pending["source_fill_index"],
                        "future_source_line": line_no,
                        "future_source_t": record.get("t"),
                        "fill_id": pending["fill_id"],
                        "venue_id": pending["venue_id"],
                        "side": pending["side"],
                        "price": pending["price"],
                        "size": pending["size"],
                        "fill_time_ms": pending["fill_time_ms"],
                        "markout_horizon_ms": horizon_ms,
                        "future_time_ms": record_time_ms,
                        "future_reference_price": future_price,
                        "future_reference_price_source": price_source,
                        "markout_pnl": markout,
                        "label_status": "OBSERVED_MARKOUT_FAIR_VALUE",
                        "label_confidence": 1.0,
                        "training_hold_reason": "requires_quote_join_holdout_and_model_calibration",
                    })
                    labels_file.write(json.dumps(markout_label, sort_keys=True, separators=(",", ":")) + "\n")
                if remaining_horizons:
                    pending["remaining_horizons_ms"] = remaining_horizons
                    still_pending.append(pending)
            pending_markouts = still_pending
        fills = record.get("fills") or []
        if not isinstance(fills, list) or not fills:
            continue
        source_t = record.get("t")
        for fill_index, fill in enumerate(fills):
            if not isinstance(fill, dict):
                continue
            fill_count += 1
            venue_id = str(fill.get("venue_id") or "unknown")
            role, role_source = _maker_taker_role(fill, lighter_trade_roles)
            per_venue[venue_id] = per_venue.get(venue_id, 0) + 1
            role_counts[role] = role_counts.get(role, 0) + 1
            venue_role_counts = role_counts_by_venue.setdefault(
                venue_id,
                {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0},
            )
            venue_role_counts[role] = venue_role_counts.get(role, 0) + 1
            markout_value = fill.get("markout_pnl_short")
            markout_status = "OBSERVED" if markout_value is not None else "MISSING"
            label_seq += 1
            fill_id = _stable_hash([
                line_no,
                source_t,
                fill_index,
                fill.get("venue_id"),
                fill.get("order_id"),
                fill.get("client_order_id"),
                fill.get("fill_time_ms"),
                fill.get("side"),
                fill.get("price"),
                fill.get("size"),
            ])
            label = _base_label(
                run_id=run_id,
                label_seq=label_seq,
                label_type="OBSERVED_FILL_LABEL",
                timestamp_ns=timestamp_ns,
            )
            label.update({
                "source": "phase5_telemetry_fills",
                "source_line": line_no,
                "source_t": source_t,
                "source_fill_index": fill_index,
                "source_record_sha256": _stable_hash(fill),
                "fill_id": fill_id,
                "venue_id": fill.get("venue_id"),
                "venue_index": fill.get("venue_index"),
                "side": fill.get("side"),
                "price": fill.get("price"),
                "size": fill.get("size"),
                "fee_bps": fill.get("fee_bps"),
                "purpose": fill.get("purpose"),
                "decision_id": fill.get("decision_id"),
                "source_decision_id": fill.get("source_decision_id"),
                "order_id_hash": _hash_or_none(fill.get("order_id")),
                "client_order_id_hash": _hash_or_none(fill.get("client_order_id")),
                "fill_time_ms": fill.get("fill_time_ms"),
                "pre_q_v": fill.get("pre_q_v"),
                "post_q_v": fill.get("post_q_v"),
                "pre_q_t": fill.get("pre_q_t"),
                "post_q_t": fill.get("post_q_t"),
                "realised_pnl_usd": fill.get("realised_pnl_usd"),
                "maker_taker_role": role,
                "maker_taker_attribution_status": "OBSERVED" if role != "UNKNOWN" else "UNKNOWN",
                "maker_taker_attribution_source": role_source,
                "markout_pnl_short": markout_value,
                "markout_status": markout_status,
                "label_status": (
                    "OBSERVED_FILL_WITH_MARKOUT"
                    if markout_status == "OBSERVED"
                    else "OBSERVED_FILL_MISSING_MARKOUT"
                ),
                "label_confidence": 1.0,
                "training_hold_reason": "requires_markout_horizons_and_holdout_join",
            })
            labels_file.write(json.dumps(label, sort_keys=True, separators=(",", ":")) + "\n")
            if markout_value is not None:
                markout_count += 1
                label_seq += 1
                markout_label = _base_label(
                    run_id=run_id,
                    label_seq=label_seq,
                    label_type="OBSERVED_MARKOUT_LABEL",
                    timestamp_ns=timestamp_ns,
                )
                markout_label.update({
                    "source": "phase5_telemetry_fills",
                    "source_line": line_no,
                    "source_t": source_t,
                    "source_fill_index": fill_index,
                    "fill_id": label["fill_id"],
                    "venue_id": fill.get("venue_id"),
                    "side": fill.get("side"),
                    "price": fill.get("price"),
                    "size": fill.get("size"),
                    "markout_horizon": "short_existing",
                    "markout_pnl": markout_value,
                    "label_status": "OBSERVED_MARKOUT_SHORT",
                    "label_confidence": 1.0,
                    "training_hold_reason": "requires_multi_horizon_markout_and_holdout_join",
                })
                labels_file.write(json.dumps(markout_label, sort_keys=True, separators=(",", ":")) + "\n")
            fill_time_ms = _safe_int(fill.get("fill_time_ms"))
            if fill_time_ms is not None and markout_horizons_ms:
                pending_markouts.append({
                    "fill_id": fill_id,
                    "source_line": line_no,
                    "source_t": source_t,
                    "source_fill_index": fill_index,
                    "venue_id": fill.get("venue_id"),
                    "venue_index": fill.get("venue_index"),
                    "side": fill.get("side"),
                    "price": fill.get("price"),
                    "size": fill.get("size"),
                    "fill_time_ms": fill_time_ms,
                    "remaining_horizons_ms": list(markout_horizons_ms),
                })
    return fill_count, markout_count, per_venue, role_counts, role_counts_by_venue


def _balance_label(
    *,
    run_id: str,
    timestamp_ns: int,
    label_seq: int,
    pre_snapshot: Path | None,
    post_snapshot: Path | None,
    comparison: Path | None,
) -> dict[str, Any] | None:
    if not comparison and not (pre_snapshot and post_snapshot):
        return None
    label = _base_label(
        run_id=run_id,
        label_seq=label_seq,
        label_type="BALANCE_RECONCILIATION_LABEL",
        timestamp_ns=timestamp_ns,
    )
    label.update({
        "label_status": "OBSERVED_BALANCE_DELTA",
        "label_confidence": 1.0,
        "source": "phase5_balance_snapshots",
        "pre_snapshot_path": str(pre_snapshot) if pre_snapshot else None,
        "post_snapshot_path": str(post_snapshot) if post_snapshot else None,
        "comparison_path": str(comparison) if comparison else None,
        "pre_snapshot_sha256": _sha256_file(pre_snapshot) if pre_snapshot else None,
        "post_snapshot_sha256": _sha256_file(post_snapshot) if post_snapshot else None,
        "comparison_sha256": _sha256_file(comparison) if comparison else None,
        "training_hold_reason": "requires_fill_markout_quote_join_before_training",
    })
    if comparison:
        data = _load_json(comparison)
        label.update({
            "balance_reconciliation_status": "OBSERVED",
            "total": data.get("total"),
            "per_venue": data.get("per_venue"),
            "venues": data.get("venues"),
            "generated_at_utc": data.get("generated_at_utc"),
        })
    else:
        pre = _load_json(pre_snapshot) if pre_snapshot else {}
        post = _load_json(post_snapshot) if post_snapshot else {}
        label.update({
            "balance_reconciliation_status": "SNAPSHOT_PAIR_OBSERVED_WITHOUT_COMPARISON",
            "pre_total_balance_usd": pre.get("total_balance_usd"),
            "post_total_balance_usd": post.get("total_balance_usd"),
            "venues": post.get("venues") or pre.get("venues"),
        })
    return label


def _gate_reason(
    fill_labels: int,
    markout_labels: int,
    balance_labels: int,
    role_counts: dict[str, int],
) -> str:
    if fill_labels == 0:
        return "observed_label_pack_missing_fills"
    if markout_labels == 0:
        return "observed_label_pack_missing_markout_coverage"
    if role_counts.get("UNKNOWN", 0) == fill_labels:
        return "observed_label_pack_missing_maker_taker_attribution"
    if role_counts.get("UNKNOWN", 0) > 0:
        return "observed_label_pack_partial_maker_taker_attribution"
    if balance_labels == 0:
        return "observed_label_pack_missing_balance_reconciliation"
    return "observed_label_pack_requires_quote_join_holdout_and_board_review"


def build_observed_labels(
    *,
    source_telemetry: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    balance_pre: Path | None,
    balance_post: Path | None,
    balance_comparison: Path | None,
    markout_horizons_ms: list[int],
    lighter_trades_json: Path | None,
) -> Path:
    if not source_telemetry.exists():
        raise ValueError(f"source telemetry not found: {source_telemetry}")
    run_id = run_id or f"PHASE51C-OBSERVED-LABELS-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    labels_path = out_dir / "labels.jsonl"
    lighter_trade_roles, lighter_trades_sha256 = _load_lighter_trade_role_index(lighter_trades_json)
    with labels_path.open("w", encoding="utf-8") as labels_file:
        fill_labels, markout_labels, per_venue_fills, role_counts, role_counts_by_venue = _extract_fill_labels(
            source_telemetry=source_telemetry,
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            markout_horizons_ms=markout_horizons_ms,
            lighter_trade_roles=lighter_trade_roles,
            labels_file=labels_file,
        )
        balance_labels = 0
        balance = _balance_label(
            run_id=run_id,
            timestamp_ns=timestamp_ns,
            label_seq=fill_labels + markout_labels + 1,
            pre_snapshot=balance_pre,
            post_snapshot=balance_post,
            comparison=balance_comparison,
        )
        if balance:
            balance_labels = 1
            labels_file.write(json.dumps(balance, sort_keys=True, separators=(",", ":")) + "\n")

    summary_path = out_dir / "observed_label_summary.json"
    summary = {
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": _gate_reason(fill_labels, markout_labels, balance_labels, role_counts),
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "source_telemetry": str(source_telemetry),
        "source_telemetry_sha256": _sha256_file(source_telemetry),
        "balance_pre_snapshot": str(balance_pre) if balance_pre else None,
        "balance_post_snapshot": str(balance_post) if balance_post else None,
        "balance_comparison": str(balance_comparison) if balance_comparison else None,
        "lighter_trades_json": str(lighter_trades_json) if lighter_trades_json else None,
        "lighter_trades_json_sha256": lighter_trades_sha256,
        "lighter_trade_role_index_size": len(lighter_trade_roles),
        "markout_horizons_ms": markout_horizons_ms,
        "fill_labels": fill_labels,
        "markout_labels": markout_labels,
        "balance_reconciliation_labels": balance_labels,
        "fill_label_status": "OBSERVED" if fill_labels else "MISSING",
        "markout_label_status": "OBSERVED" if markout_labels else "MISSING",
        "balance_reconciliation_status": "OBSERVED" if balance_labels else "MISSING",
        "per_venue_fill_counts": per_venue_fills,
        "maker_taker_role_counts": role_counts,
        "maker_taker_role_counts_by_venue": role_counts_by_venue,
        "record_count": fill_labels + markout_labels + balance_labels,
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
    parser.add_argument("--source-telemetry", type=Path, required=True)
    parser.add_argument("--balance-pre", type=Path, default=None)
    parser.add_argument("--balance-post", type=Path, default=None)
    parser.add_argument("--balance-comparison", type=Path, default=None)
    parser.add_argument("--lighter-trades-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument(
        "--markout-horizons-ms",
        default="100,500,1000,5000",
        help="Comma-separated fair-value markout horizons to derive from future telemetry records.",
    )
    args = parser.parse_args()
    try:
        markout_horizons_ms = [
            int(part)
            for part in str(args.markout_horizons_ms).split(",")
            if part.strip()
        ]
    except ValueError as exc:
        print(f"phase51c_observed_labels: ERROR: invalid --markout-horizons-ms: {exc}", file=sys.stderr)
        return 2
    try:
        out_dir = build_observed_labels(
            source_telemetry=args.source_telemetry,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            balance_pre=args.balance_pre,
            balance_post=args.balance_post,
            balance_comparison=args.balance_comparison,
            markout_horizons_ms=markout_horizons_ms,
            lighter_trades_json=args.lighter_trades_json,
        )
    except Exception as exc:
        print(f"phase51c_observed_labels: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51c_observed_labels: wrote {out_dir}")
    print("phase51c_observed_labels: status HOLD (observed evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
