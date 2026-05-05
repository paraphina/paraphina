#!/usr/bin/env python3
"""Build Phase 5.1p Lighter canonical native maker/taker role evidence.

This HOLD-only tool joins canonical observed P_fill labels to Lighter native
trade backfill rows using hashed order/client identifiers. Raw native IDs are
read only inside this quarantined process and are never emitted.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51p_lighter_native_role_canonical_join"
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
    "ask_id",
    "bid_id",
    "ask_client_id",
    "bid_client_id",
    "trade_id",
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


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in value.lower())


def _hash_or_none(value: Any) -> str | None:
    return _stable_hash(value) if value not in (None, "") else None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _role_counts(value: Any) -> dict[str, int]:
    counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
    if not isinstance(value, dict):
        return counts
    for key in counts:
        counts[key] = max(0, _safe_int(value.get(key)) or 0)
    return counts


def _known_count(counts: dict[str, int]) -> int:
    return int(counts.get("MAKER") or 0) + int(counts.get("TAKER") or 0)


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _base_record(run_id: str, seq: int, timestamp_ns: int, label_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": label_type,
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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
        "raw_identifier_redaction_status": "PASS",
    }


def _extract_items(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _id_hashes(*values: Any, prehashed_values: tuple[Any, ...] = ()) -> set[str]:
    hashes: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        hashes.add(_stable_hash(value))
        hashes.add(_stable_hash(str(value)))
    for value in prehashed_values:
        if value in (None, "") or not _is_sha256_hex(value):
            continue
        hashes.add(str(value).lower())
    return hashes


def _trade_uid_hash(trade: dict[str, Any]) -> str:
    return _stable_hash([
        trade.get("trade_id"),
        trade.get("trade_id_str"),
        trade.get("tx_hash"),
        trade.get("market_id"),
        trade.get("timestamp"),
        trade.get("transaction_time"),
        trade.get("price"),
        trade.get("size"),
        trade.get("ask_id"),
        trade.get("bid_id"),
        trade.get("ask_client_id"),
        trade.get("bid_client_id"),
        trade.get("trade_id_sha256"),
        trade.get("trade_id_str_sha256"),
        trade.get("tx_hash_sha256"),
        trade.get("ask_id_sha256"),
        trade.get("bid_id_sha256"),
        trade.get("ask_client_id_sha256"),
        trade.get("bid_client_id_sha256"),
    ])


def _native_role_for_account(trade: dict[str, Any], account_index: int | None) -> tuple[str | None, str | None, str | None]:
    if account_index is None:
        return None, None, "NATIVE_ACCOUNT_INDEX_MISSING"
    is_maker_ask = trade.get("is_maker_ask")
    if is_maker_ask is None:
        is_maker_ask = trade.get("isMakerAsk")
    if not isinstance(is_maker_ask, bool):
        return None, None, "NATIVE_ACCOUNT_ROLE_UNDERIVABLE"
    ask_account = _safe_int(trade.get("ask_account_id") or trade.get("askAccountId"))
    bid_account = _safe_int(trade.get("bid_account_id") or trade.get("bidAccountId"))
    if account_index == ask_account:
        return ("MAKER" if is_maker_ask else "TAKER"), "ASK", None
    if account_index == bid_account:
        return ("TAKER" if is_maker_ask else "MAKER"), "BID", None
    return None, None, "NATIVE_ACCOUNT_ROLE_UNDERIVABLE"


def _side_hashes(trade: dict[str, Any], account_side: str | None) -> set[str]:
    if account_side == "ASK":
        return _id_hashes(
            trade.get("ask_id"),
            trade.get("ask_id_str"),
            trade.get("ask_client_id"),
            trade.get("ask_client_id_str"),
            prehashed_values=(
                trade.get("ask_id_sha256"),
                trade.get("ask_id_str_sha256"),
                trade.get("ask_client_id_sha256"),
                trade.get("ask_client_id_str_sha256"),
            ),
        )
    if account_side == "BID":
        return _id_hashes(
            trade.get("bid_id"),
            trade.get("bid_id_str"),
            trade.get("bid_client_id"),
            trade.get("bid_client_id_str"),
            prehashed_values=(
                trade.get("bid_id_sha256"),
                trade.get("bid_id_str_sha256"),
                trade.get("bid_client_id_sha256"),
                trade.get("bid_client_id_str_sha256"),
            ),
        )
    return set()


def _load_hold_summary(run_dir: Path, filename: str) -> dict[str, Any]:
    summary_path = run_dir / filename
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    return summary


def _load_inventory(native_role_inventory_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = _load_hold_summary(native_role_inventory_run, "native_role_source_inventory_summary.json")
    labels_path = native_role_inventory_run / "native_role_source_inventory_labels.jsonl"
    labels: dict[str, dict[str, Any]] = {}
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "PHASE51O_NATIVE_ROLE_SOURCE_INVENTORY_LABEL":
            continue
        _check_unsafe(label, labels_path, label="inventory label")
        canonical_group_id = str(label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError(f"{labels_path} row missing canonical_group_id")
        labels[canonical_group_id] = label
    expected = int(summary.get("label_count") or 0)
    if len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary label_count {expected}")
    return summary, labels


def _load_canonical_pfill(canonical_pfill_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = _load_hold_summary(canonical_pfill_run, "pfill_outcome_summary.json")
    labels_path = canonical_pfill_run / "pfill_order_labels.jsonl"
    labels: dict[str, dict[str, Any]] = {}
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(label, labels_path, label="canonical pfill label")
        canonical_group_id = str(label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError(f"{labels_path} row missing canonical_group_id")
        labels[canonical_group_id] = label
    expected = int(summary.get("order_label_count") or 0)
    if len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary order_label_count {expected}")
    return summary, labels


def _source_run_paths_from_canonical(labels: dict[str, dict[str, Any]]) -> set[Path]:
    paths: set[Path] = set()
    for label in labels.values():
        for value in label.get("source_pfill_run_paths") or []:
            if value:
                paths.add(_resolve_path(Path(str(value))))
    return paths


def _load_source_pfill_labels(source_pfill_runs: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_order_key: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for run in source_pfill_runs:
        run = _resolve_path(run)
        summary = _load_hold_summary(run, "pfill_outcome_summary.json")
        summaries.append({
            "run_path": str(run),
            "run_id": summary.get("run_id"),
            "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
            "pfill_summary_sha256": _sha256_file(run / "pfill_outcome_summary.json"),
            "pfill_labels_sha256": _sha256_file(run / "pfill_order_labels.jsonl"),
        })
        for _, label in _iter_jsonl(run / "pfill_order_labels.jsonl"):
            if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
                continue
            _check_unsafe(label, run / "pfill_order_labels.jsonl", label="source pfill label")
            order_key = str(label.get("order_key") or "")
            if not order_key:
                raise ValueError(f"{run} source P_fill row missing order_key")
            if order_key in by_order_key:
                raise ValueError(f"duplicate source order_key across source P_fill runs: {order_key}")
            by_order_key[order_key] = label
    return by_order_key, summaries


def _load_native_trade_index(lighter_trade_backfill_runs: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    seen_trade_uids: set[str] = set()
    for run in lighter_trade_backfill_runs:
        run = _resolve_path(run)
        summary = _load_hold_summary(run, "lighter_trade_backfill_summary.json")
        trades_path = Path(str(summary.get("trades_path") or ""))
        if not trades_path.is_absolute():
            trades_path = ROOT / trades_path
        payload = _load_json(trades_path)
        trades = _extract_items(payload, {"trades", "trade_history", "tradehistory"})
        account_index = _safe_int(summary.get("account_index"))
        if account_index is None and isinstance(payload, dict):
            account_index = _safe_int(payload.get("account_index"))
        indexed = 0
        run_unique_indexed = 0
        run_global_duplicate_count = 0
        role_counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
        for trade in trades:
            trade_uid = _trade_uid_hash(trade)
            if trade_uid in seen_trade_uids:
                run_global_duplicate_count += 1
                continue
            seen_trade_uids.add(trade_uid)
            run_unique_indexed += 1
            role, account_side, role_error = _native_role_for_account(trade, account_index)
            role_counts[role or "UNKNOWN"] = role_counts.get(role or "UNKNOWN", 0) + 1
            if role_error or role not in {"MAKER", "TAKER"}:
                continue
            trade_record = {
                "native_trade_uid_hash": trade_uid,
                "native_role": role,
                "native_account_side": account_side,
                "native_trade_timestamp_ms": _safe_int(
                    trade.get("timestamp") or trade.get("time") or trade.get("created_at")
                ),
                "native_trade_hash": (
                    str(trade.get("tx_hash_sha256")).lower()
                    if _is_sha256_hex(trade.get("tx_hash_sha256"))
                    else _hash_or_none(trade.get("tx_hash"))
                ),
                "lighter_trade_backfill_run": str(run),
            }
            for identity_hash in _side_hashes(trade, account_side):
                by_hash.setdefault(identity_hash, []).append(trade_record)
                indexed += 1
        summaries.append({
            "run_path": str(run),
            "run_id": summary.get("run_id"),
            "lighter_trade_backfill_summary_sha256": _sha256_file(run / "lighter_trade_backfill_summary.json"),
            "trades_path_sha256": _sha256_file(trades_path),
            "account_index_present": account_index is not None,
            "native_trade_count": len(trades),
            "native_trade_unique_indexed_count": run_unique_indexed,
            "native_trade_global_duplicate_count": run_global_duplicate_count,
            "native_trade_side_identity_hash_index_entry_count": indexed,
            "native_role_counts": dict(sorted(role_counts.items())),
        })
    return by_hash, summaries


def _candidate_hashes(source_labels: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for label in source_labels:
        for field in ("order_id_hash", "client_order_id_hash"):
            value = label.get(field)
            if value:
                hashes.add(str(value))
    return hashes


def _scan_no_raw_identifiers(records: list[dict[str, Any]], *, label: str) -> None:
    for index, record in enumerate(records, start=1):
        present = sorted(field for field in RAW_IDENTIFIER_FIELDS if field in record)
        if present:
            raise ValueError(f"{label} record {index} contains raw identifier field(s): {', '.join(present)}")


def build_lighter_native_role_canonical_join(
    *,
    native_role_inventory_run: Path,
    canonical_pfill_run: Path,
    source_pfill_run: list[Path],
    lighter_trade_backfill_run: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    if not lighter_trade_backfill_run:
        raise ValueError("at least one --lighter-trade-backfill-run is required")
    run_id = run_id or f"PHASE51P-LIGHTER-NATIVE-ROLE-CANONICAL-JOIN-{_utc_stamp()}"
    output_root = _resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    native_role_inventory_run = _resolve_path(native_role_inventory_run)
    canonical_pfill_run = _resolve_path(canonical_pfill_run)
    inventory_summary, inventory_by_group = _load_inventory(native_role_inventory_run)
    canonical_summary, canonical_by_group = _load_canonical_pfill(canonical_pfill_run)
    source_paths = sorted(set(_resolve_path(path) for path in source_pfill_run) | _source_run_paths_from_canonical(canonical_by_group))
    source_by_order_key, source_summaries = _load_source_pfill_labels(source_paths)
    native_by_hash, trade_summaries = _load_native_trade_index(lighter_trade_backfill_run)

    labels: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    evidence_seq = 0
    for seq, (canonical_group_id, inventory_label) in enumerate(sorted(inventory_by_group.items()), start=1):
        canonical_label = canonical_by_group.get(canonical_group_id)
        if canonical_label is None:
            raise ValueError(f"inventory canonical_group_id missing canonical P_fill label: {canonical_group_id}")
        venue = str(canonical_label.get("venue_id") or "").lower()
        fill_count = _safe_int(canonical_label.get("fill_count")) or 0
        input_counts = _role_counts(canonical_label.get("maker_taker_role_counts"))
        inventory_status = str(inventory_label.get("native_role_source_status") or "UNKNOWN")
        status = "NOT_TARGETED"
        hold_reason = "not_lighter_source_available_no_canonical_join"
        effective_counts = input_counts
        candidate_hash_status = "NOT_EVALUATED"
        matched_trades: dict[str, dict[str, Any]] = {}
        matched_hash_count = 0
        source_order_keys = [str(value) for value in (canonical_label.get("source_order_keys") or []) if value]
        source_labels = [source_by_order_key[key] for key in source_order_keys if key in source_by_order_key]
        missing_source_label_count = len(source_order_keys) - len(source_labels)

        if venue == "lighter" and fill_count > 0 and inventory_status == "SOURCE_AVAILABLE_NO_CANONICAL_JOIN":
            if missing_source_label_count > 0:
                status = "MISSING_SOURCE_PFILL_LABEL"
                hold_reason = "source_order_key_missing_from_source_pfill_inputs"
            else:
                candidate_hashes = _candidate_hashes(source_labels)
                candidate_hash_status = "PRESENT" if candidate_hashes else "MISSING"
                ambiguous_hashes = [
                    identity_hash
                    for identity_hash in candidate_hashes
                    if len({row["native_trade_uid_hash"] for row in native_by_hash.get(identity_hash, [])}) > 1
                ]
                if not candidate_hashes:
                    status = "MISSING_SOURCE_ID_HASH"
                    hold_reason = "source_pfill_labels_lack_order_or_client_hash"
                elif ambiguous_hashes:
                    status = "AMBIGUOUS_NATIVE_ID_HASH"
                    hold_reason = "source_order_or_client_hash_matches_multiple_native_trades"
                else:
                    for identity_hash in candidate_hashes:
                        matches = native_by_hash.get(identity_hash, [])
                        matched_hash_count += 1 if matches else 0
                        for match in matches:
                            matched_trades[match["native_trade_uid_hash"]] = match
                    role_counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
                    for match in matched_trades.values():
                        role_counts[match["native_role"]] = role_counts.get(match["native_role"], 0) + 1
                    if not matched_trades:
                        status = "NATIVE_ID_HASH_NO_MATCH"
                        hold_reason = "no_lighter_native_trade_side_id_matches_source_order_hash"
                    elif len(matched_trades) < fill_count:
                        status = "PARTIAL_NATIVE_ID_HASH_MATCH"
                        hold_reason = "fewer_unique_native_trades_than_canonical_fill_count"
                        effective_counts = role_counts
                    elif len(matched_trades) > fill_count:
                        status = "NATIVE_ID_HASH_COUNT_MISMATCH"
                        hold_reason = "more_unique_native_trades_than_canonical_fill_count"
                        effective_counts = role_counts
                    elif _known_count(role_counts) >= fill_count and int(role_counts.get("UNKNOWN") or 0) == 0:
                        status = "RECOVERED_LIGHTER_NATIVE_ROLE"
                        hold_reason = "exact_hashed_lighter_native_side_id_match"
                        effective_counts = role_counts
                        evidence_seq += 1
                        evidence_record = _base_record(
                            run_id,
                            evidence_seq,
                            timestamp_ns,
                            "PHASE51P_LIGHTER_NATIVE_ROLE_EVIDENCE",
                        )
                        evidence_record.update({
                            "source": "phase51p_lighter_native_role_canonical_join",
                            "canonical_group_id": canonical_group_id,
                            "source_telemetry_sha256": canonical_label.get("source_telemetry_sha256"),
                            "venue_id": "lighter",
                            "side": canonical_label.get("side"),
                            "fill_count": fill_count,
                            "maker_taker_role_counts": role_counts,
                            "maker_taker_attribution_source": "LIGHTER_TRADES_JSON",
                            "native_role_source_status": status,
                            "native_role_source_hold_reason": hold_reason,
                            "native_trade_uid_hashes": sorted(matched_trades),
                        })
                        evidence.append(evidence_record)
                    else:
                        status = "NATIVE_ROLE_INCOMPLETE"
                        hold_reason = "native_trade_match_role_counts_incomplete"
                        effective_counts = role_counts
                if status == "MISSING_SOURCE_ID_HASH":
                    candidate_hash_status = "MISSING"

        record = _base_record(run_id, seq, timestamp_ns, "PHASE51P_LIGHTER_NATIVE_ROLE_CANONICAL_JOIN_LABEL")
        record.update({
            "source": "phase51p_lighter_native_role_canonical_join",
            "native_role_inventory_run": str(native_role_inventory_run),
            "canonical_pfill_run": str(canonical_pfill_run),
            "canonical_group_id": canonical_group_id,
            "canonical_order_key": canonical_label.get("order_key"),
            "source_telemetry_sha256": canonical_label.get("source_telemetry_sha256"),
            "venue_id": canonical_label.get("venue_id"),
            "side": canonical_label.get("side"),
            "fill_count": fill_count,
            "input_maker_taker_role_counts": input_counts,
            "effective_maker_taker_role_counts": effective_counts,
            "inventory_native_role_source_status": inventory_status,
            "lighter_native_role_join_status": status,
            "lighter_native_role_join_hold_reason": hold_reason,
            "source_order_key_count": len(source_order_keys),
            "source_pfill_label_count": len(source_labels),
            "missing_source_pfill_label_count": missing_source_label_count,
            "source_identity_hash_status": candidate_hash_status,
            "source_identity_hash_count": len(_candidate_hashes(source_labels)),
            "matched_identity_hash_count": matched_hash_count,
            "matched_native_trade_count": len(matched_trades),
            "matched_native_trade_uid_hashes": sorted(matched_trades),
            "maker_taker_attribution_source": (
                "LIGHTER_TRADES_JSON" if status == "RECOVERED_LIGHTER_NATIVE_ROLE" else "UNRECOVERED"
            ),
            "native_role_evidence_record_emitted": status == "RECOVERED_LIGHTER_NATIVE_ROLE",
        })
        labels.append(record)

    _scan_no_raw_identifiers(labels, label="join label")
    _scan_no_raw_identifiers(evidence, label="evidence")
    labels_path = out_dir / "lighter_native_role_canonical_join_labels.jsonl"
    evidence_path = out_dir / "lighter_native_role_evidence.jsonl"
    summary_path = out_dir / "lighter_native_role_canonical_join_summary.json"
    _write_jsonl(labels_path, labels)
    _write_jsonl(evidence_path, evidence)

    status_counts = _status_counts(labels, "lighter_native_role_join_status")
    lighter_source_available_target_count = sum(
        1
        for record in labels
        if record.get("inventory_native_role_source_status") == "SOURCE_AVAILABLE_NO_CANONICAL_JOIN"
        and str(record.get("venue_id") or "").lower() == "lighter"
        and int(record.get("fill_count") or 0) > 0
    )
    recovered_lighter_native_role_count = status_counts.get("RECOVERED_LIGHTER_NATIVE_ROLE", 0)
    gate_reason = (
        "phase51p_lighter_native_role_join_complete_nonlive_hold"
        if lighter_source_available_target_count == recovered_lighter_native_role_count
        else "phase51p_lighter_native_role_join_incomplete"
    )
    summary = {
        "schema_version": 1,
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
        "raw_identifier_redaction_status": "PASS",
        "native_role_inventory_run": str(native_role_inventory_run),
        "native_role_inventory_summary_sha256": _sha256_file(
            native_role_inventory_run / "native_role_source_inventory_summary.json"
        ),
        "canonical_pfill_run": str(canonical_pfill_run),
        "canonical_pfill_summary_sha256": _sha256_file(canonical_pfill_run / "pfill_outcome_summary.json"),
        "canonical_pfill_labels_sha256": _sha256_file(canonical_pfill_run / "pfill_order_labels.jsonl"),
        "source_pfill_inputs": source_summaries,
        "lighter_trade_backfill_inputs": trade_summaries,
        "label_count": len(labels),
        "filled_count": sum(1 for record in labels if int(record.get("fill_count") or 0) > 0),
        "lighter_source_available_target_count": lighter_source_available_target_count,
        "recovered_lighter_native_role_count": recovered_lighter_native_role_count,
        "unrecovered_lighter_native_role_count": (
            lighter_source_available_target_count - recovered_lighter_native_role_count
        ),
        "native_role_evidence_record_count": len(evidence),
        "lighter_native_role_join_status_counts": status_counts,
        "native_role_inventory_gate_reason": inventory_summary.get("gate_reason"),
        "canonical_pfill_gate_reason": canonical_summary.get("gate_reason"),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, evidence_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, evidence_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-role-inventory-run", type=Path, required=True)
    parser.add_argument("--canonical-pfill-run", type=Path, required=True)
    parser.add_argument("--source-pfill-run", type=Path, action="append", default=[])
    parser.add_argument("--lighter-trade-backfill-run", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_lighter_native_role_canonical_join(
            native_role_inventory_run=args.native_role_inventory_run,
            canonical_pfill_run=args.canonical_pfill_run,
            source_pfill_run=args.source_pfill_run,
            lighter_trade_backfill_run=args.lighter_trade_backfill_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51p_lighter_native_role_canonical_join: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51p_lighter_native_role_canonical_join: wrote {out_dir}")
    print("phase51p_lighter_native_role_canonical_join: status HOLD (Lighter native role evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
