#!/usr/bin/env python3
"""Acquire Phase 5.1r forward venue-native source evidence.

This HOLD-only tool normalizes local, read-only native source snapshots into
the sanitized JSONL inputs consumed by Phase 5.1q. Raw venue identifiers may be
present in quarantined input snapshots, but they are never emitted.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51r_forward_native_source_acquisition"
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
    "fill_id",
    "id",
    "oid",
    "cloid",
    "tid",
    "orderId",
    "clientOrderId",
    "tradeId",
    "fillId",
    "order_id_str",
    "client_id",
    "clientId",
}
NESTED_RAW_IDENTIFIER_FIELDS = RAW_IDENTIFIER_FIELDS | {"i"}
SECRET_FIELD_FRAGMENTS = {
    "api_key",
    "apikey",
    "private_key",
    "privatekey",
    "secret_key",
    "secretkey",
    "access_key",
    "accesskey",
    "auth_token",
    "authtoken",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "session_token",
    "signing_key",
}
SOURCE_LINK_HASH_FIELDS = {
    "phase51s_source_record_sha256",
    "source_record_sha256",
    "redacted_source_record_sha256",
}
SOURCE_LINK_ALLOWED_FIELDS = SOURCE_LINK_HASH_FIELDS | {"canonical_group_id", "order_key"} | UNSAFE_TRUE_FLAGS
ROLE_VALUES = {"MAKER", "TAKER"}
SOURCE_BY_VENUE = {
    "hyperliquid": "HYPERLIQUID_CROSSED",
    "paradex": "PARADEX_LIQUIDITY",
    "aster": "ASTER_ORDER_TRADE_UPDATE_M",
    "extended": "EXTENDED_ISTAKER",
    "lighter": "LIGHTER_TRADES_JSON",
}
SOURCE_LIST_KEYS = {
    "data",
    "events",
    "fills",
    "results",
    "rows",
    "trade_history",
    "tradeHistory",
    "trades",
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
            _check_output_safe(record, path)
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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> bool:
    parsed = _safe_float(value)
    return parsed is not None and parsed > 0.0


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _check_unsafe_flags(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _field_looks_secret(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if "nonsecret" in normalized:
        return False
    if "authorization" in normalized and normalized.startswith(("no_", "not_")):
        return False
    return any(fragment in normalized for fragment in SECRET_FIELD_FRAGMENTS)


def _check_no_secret_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_no_nested_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = NESTED_RAW_IDENTIFIER_FIELDS & set(obj)
        if raw_fields:
            raise ValueError(f"{path} {label} leaked raw identifier fields: {sorted(raw_fields)}")


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    raw_fields = RAW_IDENTIFIER_FIELDS & set(record)
    if raw_fields:
        raise ValueError(f"{path} output leaked raw identifier fields: {sorted(raw_fields)}")


def _check_source_link_safe(row: dict[str, Any], path: Path, line_no: int) -> None:
    _check_unsafe_flags(row, path, label="source link")
    _check_no_secret_fields(row, path, label="source link")
    _check_no_nested_raw_identifier_fields(row, path, label="source link")
    unexpected = sorted(set(row) - SOURCE_LINK_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"{path}:{line_no} source link has unsupported fields: {unexpected}")


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


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


def _role_counts(role: str | None) -> dict[str, int]:
    counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
    if role in ROLE_VALUES:
        counts[role] = 1
    else:
        counts["UNKNOWN"] = 1
    return counts


def _known_count(counts: dict[str, int]) -> int:
    return int(counts.get("MAKER") or 0) + int(counts.get("TAKER") or 0)


def _fill_count(label: dict[str, Any] | None) -> int:
    return _safe_int((label or {}).get("fill_count")) or 0


def _existing_role_counts(value: Any) -> dict[str, int]:
    counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
    if not isinstance(value, dict):
        return counts
    for key in counts:
        counts[key] = max(0, _safe_int(value.get(key)) or 0)
    return counts


def _merge_role_counts(existing: dict[str, int], additional: dict[str, int]) -> dict[str, int]:
    return {
        key: int(existing.get(key) or 0) + int(additional.get(key) or 0)
        for key in ("MAKER", "TAKER", "UNKNOWN")
    }


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _load_observed_pfill(observed_pfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = observed_pfill_run / "pfill_outcome_summary.json"
    labels_path = observed_pfill_run / "pfill_order_labels.jsonl"
    summary = _load_json(summary_path)
    _check_unsafe_flags(summary, summary_path, label="summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe_flags(label, labels_path, label="pfill label")
        labels.append(label)
    expected = _safe_int(summary.get("order_label_count"))
    if expected is not None and len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary order_label_count {expected}")
    return summary, labels


def _source_paths(source_roots: list[Path], source_json: list[Path]) -> list[Path]:
    paths = [_resolve_path(path) for path in source_json]
    for root in source_roots:
        root = _resolve_path(root)
        if root.is_file():
            paths.append(root)
            continue
        if not root.exists():
            raise ValueError(f"source root does not exist: {root}")
        for suffix in ("*.json", "*.jsonl"):
            paths.extend(sorted(root.rglob(suffix)))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _top_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "account_index",
            "canonical_group_id",
            "market",
            "market_id",
            "market_symbol",
            "order_key",
            "venue",
            "venue_id",
        )
        if key in record
    }


def _payload_records(payload: Any, inherited: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    inherited = dict(inherited or {})
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_payload_records(item, inherited))
        return out
    if not isinstance(payload, dict):
        return []
    merged_inherited = {**inherited, **_top_metadata(payload)}
    for key in SOURCE_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            out: list[dict[str, Any]] = []
            for item in value:
                out.extend(_payload_records(item, merged_inherited))
            return out
    return [{**inherited, **payload}]


def _iter_source_records(path: Path):
    if path.suffix == ".jsonl":
        for line_no, row in _iter_jsonl(path):
            _check_unsafe_flags(row, path, label="source row")
            for item in _payload_records(row):
                yield line_no, item
        return
    payload = _load_json(path)
    _check_unsafe_flags(payload, path, label="source payload")
    for idx, row in enumerate(_payload_records(payload), start=1):
        if isinstance(row, dict):
            _check_unsafe_flags(row, path, label="source row")
            yield idx, row


def _venue_id(row: dict[str, Any], observed_label: dict[str, Any] | None = None) -> str:
    raw = row.get("venue_id") or row.get("venue") or (observed_label or {}).get("venue_id") or ""
    return str(raw).lower()


def _source_link_hashes(row: dict[str, Any]) -> list[str]:
    hashes = []
    for key in ("phase51s_source_record_sha256", "source_record_sha256", "redacted_source_record_sha256"):
        value = str(row.get(key) or "")
        if value:
            hashes.append(value)
    hashes.append(_stable_hash(row))
    out: list[str] = []
    seen: set[str] = set()
    for value in hashes:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_link_group(
    row: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
    path: Path,
    line_no: int,
) -> str:
    group = str(row.get("canonical_group_id") or "")
    order_key = str(row.get("order_key") or "")
    resolved_from_order = ""
    if order_key:
        if order_key not in by_order_key:
            raise ValueError(f"{path}:{line_no} source link order_key does not match observed P-fill labels")
        resolved_from_order = str(by_order_key[order_key].get("canonical_group_id") or "")
    if group:
        if group not in by_group:
            raise ValueError(f"{path}:{line_no} source link canonical_group_id does not match observed P-fill labels")
        if resolved_from_order and resolved_from_order != group:
            raise ValueError(f"{path}:{line_no} source link canonical_group_id conflicts with order_key")
        return group
    if resolved_from_order:
        return resolved_from_order
    raise ValueError(f"{path}:{line_no} source link missing canonical_group_id or order_key")


def _load_source_links(
    paths: list[Path],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    links: dict[str, dict[str, Any]] = {}
    infos: list[dict[str, Any]] = []
    total_count = 0
    for raw_path in paths:
        path = _resolve_path(raw_path)
        count = 0
        for line_no, row in _iter_jsonl(path):
            _check_source_link_safe(row, path, line_no)
            group = _resolve_link_group(row, by_group, by_order_key, path, line_no)
            source_hashes = [
                str(row.get(key) or "")
                for key in SOURCE_LINK_HASH_FIELDS
                if row.get(key)
            ]
            if not source_hashes:
                raise ValueError(f"{path}:{line_no} source link missing source hash")
            for source_hash in sorted(set(source_hashes)):
                existing = links.get(source_hash)
                if existing is not None:
                    raise ValueError(f"{path}:{line_no} duplicate source link hash {source_hash}")
                links[source_hash] = {
                    "canonical_group_id": group,
                    "order_key": row.get("order_key") or by_group[group].get("order_key"),
                    "source_link_sha256": _stable_hash(row),
                }
            count += 1
            total_count += 1
        infos.append({"path": str(path), "sha256": _sha256_file(path), "record_count": count})
    return links, infos, total_count


def _canonical_group_with_source(
    row: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
    source_links: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    group = str(row.get("canonical_group_id") or "")
    if group in by_group:
        return group, "SOURCE_ROW_CANONICAL_GROUP"
    order_key = str(row.get("order_key") or "")
    if order_key in by_order_key:
        return str(by_order_key[order_key].get("canonical_group_id") or ""), "SOURCE_ROW_ORDER_KEY"
    linked_groups = {
        str(source_links[source_hash].get("canonical_group_id") or "")
        for source_hash in _source_link_hashes(row)
        if source_hash in source_links
    }
    if len(linked_groups) > 1:
        raise ValueError("source row has ambiguous source-link sidecar groups")
    if linked_groups:
        return linked_groups.pop(), "SOURCE_LINK_SIDECAR"
    return None, "NO_CANONICAL_LINK"


def _canonical_group(row: dict[str, Any], by_group: dict[str, dict[str, Any]], by_order_key: dict[str, dict[str, Any]]) -> str | None:
    group, _ = _canonical_group_with_source(row, by_group, by_order_key, {})
    return group


def _native_role_for_lighter_account(row: dict[str, Any]) -> tuple[str | None, str | None]:
    account_index = _safe_int(row.get("account_index"))
    if account_index is None:
        return None, "lighter_account_index_missing"
    is_maker_ask = row.get("is_maker_ask")
    if is_maker_ask is None:
        is_maker_ask = row.get("isMakerAsk")
    if not isinstance(is_maker_ask, bool):
        return None, "lighter_is_maker_ask_missing"
    ask_account = _safe_int(row.get("ask_account_id") or row.get("askAccountId"))
    bid_account = _safe_int(row.get("bid_account_id") or row.get("bidAccountId"))
    if account_index == ask_account:
        return ("MAKER" if is_maker_ask else "TAKER"), None
    if account_index == bid_account:
        return ("TAKER" if is_maker_ask else "MAKER"), None
    return None, "lighter_account_side_unmatched"


def _detect_native_role(row: dict[str, Any], venue: str) -> tuple[str | None, str | None, str | None]:
    explicit_role = str(row.get("native_role") or row.get("native_liquidity_role") or "").upper()
    explicit_source = str(row.get("maker_taker_attribution_source") or row.get("native_role_source") or "")
    if explicit_role in ROLE_VALUES:
        if explicit_source in {"VENUE_NATIVE_FILL_FIELD", "VENUE_NATIVE_TRADE_JOIN", "VENUE_NATIVE_FEE_ROLE"}:
            return explicit_role, explicit_source, None
        return None, "VENUE_NATIVE_FILL_FIELD", "generic_native_role_missing_native_provenance"
    if venue == "hyperliquid" or (not venue and "crossed" in row):
        crossed = row.get("crossed")
        if isinstance(crossed, bool):
            return ("TAKER" if crossed else "MAKER"), "HYPERLIQUID_CROSSED", None
        return None, "HYPERLIQUID_CROSSED", "hyperliquid_crossed_missing"
    if venue == "paradex" or (not venue and "liquidity" in row):
        liquidity = str(row.get("liquidity") or "").upper()
        if liquidity in ROLE_VALUES:
            return liquidity, "PARADEX_LIQUIDITY", None
        return None, "PARADEX_LIQUIDITY", "paradex_liquidity_missing"
    if venue == "extended" or (not venue and ("isTaker" in row or "is_taker" in row)):
        is_taker = row.get("isTaker")
        if is_taker is None:
            is_taker = row.get("is_taker")
        if isinstance(is_taker, bool):
            return ("TAKER" if is_taker else "MAKER"), "EXTENDED_ISTAKER", None
        return None, "EXTENDED_ISTAKER", "extended_is_taker_missing"
    if venue == "aster" or (not venue and row.get("e") == "ORDER_TRADE_UPDATE"):
        order_update = row.get("o") if isinstance(row.get("o"), dict) else row
        maker_side = order_update.get("m")
        fill_qty_present = _positive_float(order_update.get("l")) or _positive_float(order_update.get("lastFilledQty"))
        if not fill_qty_present:
            return None, "ASTER_ORDER_TRADE_UPDATE_M", "aster_no_trade_fill_quantity"
        if isinstance(maker_side, bool):
            return ("MAKER" if maker_side else "TAKER"), "ASTER_ORDER_TRADE_UPDATE_M", None
        return None, "ASTER_ORDER_TRADE_UPDATE_M", "aster_maker_side_missing"
    if venue == "lighter":
        role, error = _native_role_for_lighter_account(row)
        return role, "LIGHTER_TRADES_JSON", error
    return None, SOURCE_BY_VENUE.get(venue), "unrecognized_native_role_source"


def _has_limit_fields(row: dict[str, Any]) -> bool:
    keys = {
        "active_order_headroom_account",
        "active_order_headroom_market",
        "sendtx_per_minute_limit",
        "sendtx_per_minute_remaining",
        "rest_requests_per_minute_limit",
        "rest_requests_per_minute_remaining",
        "weighted_requests_per_minute_limit",
        "weighted_requests_per_minute_remaining",
        "native_limit_event_time_status",
    }
    return any(key in row for key in keys)


def _limit_record(row: dict[str, Any], group: str, venue: str, run_id: str, seq: int, timestamp_ns: int) -> dict[str, Any]:
    record = _base_record(run_id, seq, timestamp_ns, "PHASE51R_NATIVE_LIMIT_SOURCE")
    record.update(
        {
            "canonical_group_id": group,
            "venue_id": venue,
            "active_order_headroom_account": _safe_int(row.get("active_order_headroom_account")),
            "active_order_headroom_market": _safe_int(row.get("active_order_headroom_market")),
            "sendtx_per_minute_limit": _safe_int(row.get("sendtx_per_minute_limit")),
            "sendtx_per_minute_remaining": _safe_int(row.get("sendtx_per_minute_remaining")),
            "rest_requests_per_minute_limit": _safe_int(row.get("rest_requests_per_minute_limit")),
            "rest_requests_per_minute_remaining": _safe_int(row.get("rest_requests_per_minute_remaining")),
            "weighted_requests_per_minute_limit": _safe_int(row.get("weighted_requests_per_minute_limit")),
            "weighted_requests_per_minute_remaining": _safe_int(row.get("weighted_requests_per_minute_remaining")),
            "native_limit_event_time_status": str(row.get("native_limit_event_time_status") or ""),
            "native_limit_staleness_ms": _safe_float(row.get("native_limit_staleness_ms")),
            "source_record_sha256": _stable_hash(row),
        }
    )
    return record


def _limit_source_complete(source: dict[str, Any]) -> bool:
    has_active = (
        source.get("active_order_headroom_account") is not None
        and source.get("active_order_headroom_market") is not None
    )
    has_sendtx = (
        source.get("sendtx_per_minute_limit") is not None
        and source.get("sendtx_per_minute_remaining") is not None
    )
    has_rest_or_weighted = (
        source.get("rest_requests_per_minute_limit") is not None
        and source.get("rest_requests_per_minute_remaining") is not None
    ) or (
        source.get("weighted_requests_per_minute_limit") is not None
        and source.get("weighted_requests_per_minute_remaining") is not None
    )
    event_time_ok = source.get("native_limit_event_time_status") in {
        "EVENT_TIME_ALIGNED",
        "SNAPSHOT_AT_DECISION_TIME",
        "OBSERVED_AT_DECISION_TIME",
    }
    return has_active and has_sendtx and has_rest_or_weighted and event_time_ok


def build_forward_native_source_acquisition(
    *,
    observed_pfill_run: Path,
    source_roots: list[Path],
    source_json: list[Path],
    source_link_jsonl: list[Path],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    observed_pfill_run = _resolve_path(observed_pfill_run)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_summary, labels = _load_observed_pfill(observed_pfill_run)
    by_group = {str(label.get("canonical_group_id") or ""): label for label in labels}
    by_order_key = {str(label.get("order_key") or ""): label for label in labels if label.get("order_key")}
    role_target_groups = {
        group
        for group, label in by_group.items()
        if (_safe_int(label.get("fill_count")) or 0) > _known_count(_existing_role_counts(label.get("maker_taker_role_counts")))
    }
    lighter_limit_target_groups = {
        group for group, label in by_group.items() if str(label.get("venue_id") or "").lower() == "lighter"
    }
    source_links, source_link_infos, source_link_record_count = _load_source_links(
        source_link_jsonl,
        by_group,
        by_order_key,
    )

    source_files = _source_paths(source_roots, source_json)
    native_roles_by_group: dict[str, dict[str, Any]] = {}
    native_role_hashes_by_group: dict[str, set[str]] = {}
    native_limits_by_group: dict[str, dict[str, Any]] = {}
    labels_out: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    seq = 0
    source_rows_seen = 0
    source_link_applied_count = 0

    for path in source_files:
        path = _resolve_path(path)
        row_count = 0
        role_count = 0
        limit_count = 0
        for line_no, row in _iter_source_records(path):
            row_count += 1
            source_rows_seen += 1
            group, link_source = _canonical_group_with_source(row, by_group, by_order_key, source_links)
            if link_source == "SOURCE_LINK_SIDECAR":
                source_link_applied_count += 1
            observed_label = by_group.get(group or "")
            venue = _venue_id(row, observed_label)
            role, role_source, role_error = _detect_native_role(row, venue)
            status = "UNJOINED_NO_CANONICAL_GROUP" if not group else "NO_NATIVE_ROLE_FIELD"
            hold_reason = "missing_canonical_group_or_order_key" if not group else (role_error or "role_not_present")
            if group and role in ROLE_VALUES and role_source:
                status = "NATIVE_ROLE_SOURCE_CAPTURED"
                hold_reason = "explicit_venue_native_role_field"
                source_record_sha = _stable_hash(row)
                native_role_hashes = native_role_hashes_by_group.setdefault(group, set())
                role_record = _base_record(run_id, seq, timestamp_ns, "PHASE51R_NATIVE_ROLE_SOURCE")
                seq += 1
                role_record.update(
                    {
                        "canonical_group_id": group,
                        "order_key": observed_label.get("order_key") if observed_label else None,
                        "venue_id": venue,
                        "maker_taker_role_counts": _role_counts(role),
                        "maker_taker_attribution_source": role_source,
                        "native_role_source_status": status,
                        "source_record_sha256": source_record_sha,
                    }
                )
                if source_record_sha not in native_role_hashes:
                    native_role_hashes.add(source_record_sha)
                    if group not in native_roles_by_group:
                        role_record["source_record_count"] = 1
                        native_roles_by_group[group] = role_record
                    else:
                        existing = native_roles_by_group[group]
                        if existing.get("venue_id") != venue:
                            raise ValueError("same canonical group mapped to conflicting venue native role sources")
                        if existing.get("maker_taker_attribution_source") != role_source:
                            raise ValueError("same canonical group mapped to conflicting native role source types")
                        existing["maker_taker_role_counts"] = _merge_role_counts(
                            _existing_role_counts(existing.get("maker_taker_role_counts")),
                            _role_counts(role),
                        )
                        existing["source_record_count"] = int(existing.get("source_record_count") or 1) + 1
                    role_count += 1

            if group and venue == "lighter" and _has_limit_fields(row):
                limit_record = _limit_record(row, group, venue, run_id, seq, timestamp_ns)
                seq += 1
                native_limits_by_group[group] = limit_record
                limit_count += 1

            label = _base_record(run_id, seq, timestamp_ns, "PHASE51R_SOURCE_ACQUISITION_LABEL")
            seq += 1
            label.update(
                {
                    "canonical_group_id": group,
                    "venue_id": venue or "unknown",
                    "source_path_hash": _stable_hash(str(path)),
                    "source_line": line_no,
                    "source_record_sha256": _stable_hash(row),
                    "canonical_group_link_source": link_source,
                    "native_source_acquisition_status": status,
                    "native_source_acquisition_hold_reason": hold_reason,
                    "maker_taker_attribution_source": role_source or "NONE",
                    "native_limit_source_captured": bool(group and venue == "lighter" and _has_limit_fields(row)),
                }
            )
            labels_out.append(label)
        source_artifacts.append(
            {
                "path_hash": _stable_hash(str(path)),
                "sha256": _sha256_file(path),
                "row_count": row_count,
                "role_count": role_count,
                "limit_count": limit_count,
            }
        )

    native_role_records = [native_roles_by_group[group] for group in sorted(native_roles_by_group)]
    native_limit_records = [native_limits_by_group[group] for group in sorted(native_limits_by_group)]

    native_role_path = out_dir / "native_role_source.jsonl"
    native_limit_path = out_dir / "native_limit_source.jsonl"
    labels_path = out_dir / "source_acquisition_labels.jsonl"
    summary_path = out_dir / "phase51r_forward_native_source_acquisition_summary.json"
    manifest_path = out_dir / "phase51r_manifest.json"

    _write_jsonl(native_role_path, native_role_records)
    _write_jsonl(native_limit_path, native_limit_records)
    _write_jsonl(labels_path, labels_out)

    complete_role_groups = {
        group
        for group, source in native_roles_by_group.items()
        if _known_count(source.get("maker_taker_role_counts") or {}) >= _fill_count(by_group.get(group))
    }
    role_recovered_targets = len(role_target_groups & complete_role_groups)
    complete_limit_groups = {group for group, source in native_limits_by_group.items() if _limit_source_complete(source)}
    limit_recovered_targets = len(lighter_limit_target_groups & complete_limit_groups)
    gate_reason = (
        "phase51r_forward_native_source_acquisition_complete_nonlive_hold"
        if role_recovered_targets == len(role_target_groups) and limit_recovered_targets == len(lighter_limit_target_groups)
        else "phase51r_forward_native_source_acquisition_incomplete"
    )
    summary = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
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
        "source_observed_pfill_run": str(observed_pfill_run),
        "source_observed_pfill_summary_sha256": _sha256_file(observed_pfill_run / "pfill_outcome_summary.json"),
        "source_observed_pfill_labels_sha256": _sha256_file(observed_pfill_run / "pfill_order_labels.jsonl"),
        "observed_pfill_label_count": len(labels),
        "source_file_count": len(source_files),
        "source_row_count": source_rows_seen,
        "source_link_record_count": source_link_record_count,
        "source_link_hash_count": len(source_links),
        "source_link_applied_count": source_link_applied_count,
        "native_role_target_count": len(role_target_groups),
        "native_role_source_record_count": len(native_role_records),
        "native_role_target_recovered_count": role_recovered_targets,
        "lighter_native_limit_target_count": len(lighter_limit_target_groups),
        "native_limit_source_record_count": len(native_limit_records),
        "native_limit_complete_source_record_count": len(complete_limit_groups),
        "lighter_native_limit_target_recovered_count": limit_recovered_targets,
        "native_source_acquisition_status_counts": _status_counts(labels_out, "native_source_acquisition_status"),
        "canonical_group_link_source_counts": _status_counts(labels_out, "canonical_group_link_source"),
        "source_artifacts": source_artifacts,
        "source_link_artifacts": source_link_infos,
        "observed_pfill_gate_status": observed_summary.get("gate_status"),
        "observed_pfill_gate_reason": observed_summary.get("gate_reason"),
    }
    _write_json(summary_path, summary)

    artifacts = [native_role_path, native_limit_path, labels_path, summary_path]
    manifest = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "artifacts": _artifact_infos(out_dir, artifacts),
    }
    _write_json(manifest_path, manifest)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", default=[])
    parser.add_argument("--source-json", type=Path, action="append", default=[])
    parser.add_argument("--source-link-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51r_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_forward_native_source_acquisition(
            observed_pfill_run=args.observed_pfill_run,
            source_roots=args.source_root,
            source_json=args.source_json,
            source_link_jsonl=args.source_link_jsonl,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51r_forward_native_source_acquisition: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51r_forward_native_source_acquisition: wrote {out_dir}")
    print("phase51r_forward_native_source_acquisition: status HOLD (read-only native source acquisition only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
