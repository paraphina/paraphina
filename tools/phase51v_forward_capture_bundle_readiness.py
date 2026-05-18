#!/usr/bin/env python3
"""Build Phase 5.1v forward capture bundle readiness artifacts.

This HOLD-only gate checks whether a local candidate source manifest is
structurally ready to feed the Phase 5.1s -> 5.1r -> 5.1q source evidence
ladder against a Phase 5.1u target manifest. It performs no network access,
reads no secrets, submits no orders, and does not infer maker/taker role or
native-limit pressure.
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

import phase51ap_lighter_pressure_sidecar_schema as pressure_schema


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51v_forward_capture_bundle_readiness"
UNSAFE_TRUE_FLAGS = {
    "approved_for_model_training",
    "approved_for_live",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_financial_claim",
    "admissible_for_model_training",
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
SOURCE_LINK_HASH_FIELDS = {
    "phase51s_source_record_sha256",
    "source_record_sha256",
    "redacted_source_record_sha256",
}
SOURCE_LINK_ALLOWED_FIELDS = SOURCE_LINK_HASH_FIELDS | {"canonical_group_id", "order_key"} | UNSAFE_TRUE_FLAGS
VENUE_REQUIRED_ROLE_FIELDS = {
    "aster": ("e_or_equivalent_order_trade_update", "aster_maker_side", "aster_positive_fill_qty"),
    "extended": ("extended_is_taker",),
    "hyperliquid": ("hyperliquid_crossed",),
    "lighter": ("lighter_account_index", "lighter_is_maker_ask", "lighter_ask_account_id", "lighter_bid_account_id"),
    "paradex": ("paradex_liquidity",),
}
LIGHTER_LIMIT_FIELD_KEYS = (
    "active_order_headroom_account",
    "active_order_headroom_market",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "rest_or_weighted_limit",
    "rest_or_weighted_remaining",
    "native_limit_event_time_status",
)
LIGHTER_LIMIT_ALIGNMENT_OK = {
    "EVENT_TIME_ALIGNED",
    "SNAPSHOT_AT_DECISION_TIME",
    "OBSERVED_AT_DECISION_TIME",
}
PRESSURE_UNAVAILABLE_STATUS = "PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD"
PRESSURE_UNAVAILABLE_REASON = "lighter_explicit_pressure_source_closed_negative"


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


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


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


def _positive_float(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _path_text_is_unsafe(path_text: str) -> bool:
    lower = path_text.lower()
    return lower.startswith(("http://", "https://")) or "://" in lower


def _path_text_is_placeholder(path_text: str) -> bool:
    stripped = path_text.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink source path is prohibited: {candidate}")


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


def _check_unsafe_flags(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


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
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
    }


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _target_group(row: dict[str, Any]) -> str:
    return str(row.get("canonical_group_id") or "")


def _target_order_key(row: dict[str, Any]) -> str:
    return str(row.get("order_key") or "")


def _target_id(row: dict[str, Any]) -> str:
    return _target_group(row) or _target_order_key(row)


def _target_maps(targets: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    by_order_key: dict[str, dict[str, Any]] = {}
    for row in targets:
        group = _target_group(row)
        order_key = _target_order_key(row)
        if group:
            by_group[group] = row
        if order_key:
            by_order_key[order_key] = row
    return by_group, by_order_key


def _resolve_target(
    row: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    group, order_key = _row_join_keys(row)
    if group and group in by_group:
        return by_group[group]
    if order_key and order_key in by_order_key:
        return by_order_key[order_key]
    return None


def _source_link_hashes(row: dict[str, Any]) -> list[str]:
    hashes = [
        str(row.get(key) or "")
        for key in SOURCE_LINK_HASH_FIELDS
        if row.get(key)
    ]
    hashes.append(_stable_hash(row))
    out: list[str] = []
    seen: set[str] = set()
    for value in hashes:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_target_strict(
    row: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
    path: Path,
    line_no: int,
    *,
    target_label: str,
) -> dict[str, Any] | None:
    group, order_key = _row_join_keys(row)
    group_target = by_group.get(group) if group else None
    order_target = by_order_key.get(order_key) if order_key else None
    if group_target is not None and order_target is not None and _target_id(group_target) != _target_id(order_target):
        raise ValueError(f"{path}:{line_no} source link {target_label} canonical_group_id conflicts with order_key")
    return group_target or order_target


def _resolve_target_with_source_link(
    row: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
    source_link_targets: dict[str, dict[str, Any]],
    *,
    target_key: str,
) -> tuple[dict[str, Any] | None, str]:
    group, order_key = _row_join_keys(row)
    if group and group in by_group:
        return by_group[group], "SOURCE_ROW_CANONICAL_GROUP"
    if order_key and order_key in by_order_key:
        return by_order_key[order_key], "SOURCE_ROW_ORDER_KEY"
    linked_targets = {
        _target_id(target): target
        for source_hash in _source_link_hashes(row)
        if source_hash in source_link_targets
        for target in [source_link_targets[source_hash].get(target_key)]
        if isinstance(target, dict)
    }
    if len(linked_targets) > 1:
        raise ValueError("source row has ambiguous source-link sidecar targets")
    if linked_targets:
        return next(iter(linked_targets.values())), "SOURCE_LINK_SIDECAR"
    return None, "NO_CANONICAL_LINK"


def _artifact_status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _load_target_run(target_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = target_run / "phase51u_forward_capture_target_manifest_summary.json"
    role_path = target_run / "native_role_capture_targets.jsonl"
    limit_path = target_run / "lighter_native_limit_capture_targets.jsonl"
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {summary_path}")
    _check_unsafe_flags(summary, summary_path, label="target summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    role_targets = [row for _, row in _iter_jsonl(role_path)]
    limit_targets = [row for _, row in _iter_jsonl(limit_path)]
    for row in role_targets:
        _check_unsafe_flags(row, role_path, label="role target")
    for row in limit_targets:
        _check_unsafe_flags(row, limit_path, label="limit target")
    return summary, role_targets, limit_targets


def _manifest_paths(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _source_status(path_text: str, manifest_path: Path, *, entry_kind: str) -> tuple[str, Path | None]:
    if not path_text.strip():
        return "MISSING_PATH_FIELD", None
    if _path_text_is_unsafe(path_text):
        raise ValueError(f"{manifest_path}: network {entry_kind} paths are prohibited")
    if _path_text_is_placeholder(path_text):
        return "PLACEHOLDER_PATH", None
    path = _resolve_path(Path(path_text))
    if _is_env_path(path):
        raise ValueError(f"{manifest_path}: env files are prohibited as native {entry_kind} input")
    _check_no_symlink(path)
    if not path.exists():
        return "MISSING_LOCAL_PATH", path
    if not path.is_file():
        return "NOT_A_FILE", path
    if path.suffix not in {".json", ".jsonl"}:
        return "UNSUPPORTED_SUFFIX", path
    return "LOCAL_FILE_READY", path


def _top_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("account_index", "canonical_group_id", "market", "market_id", "order_key", "venue", "venue_id")
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
            _check_no_secret_fields(row, path, label="source row")
            _check_no_nested_raw_identifier_fields(row, path, label="source row")
            for item in _payload_records(row):
                yield line_no, item
        return
    payload = _load_json(path)
    _check_unsafe_flags(payload, path, label="source payload")
    _check_no_secret_fields(payload, path, label="source payload")
    _check_no_nested_raw_identifier_fields(payload, path, label="source payload")
    for idx, row in enumerate(_payload_records(payload), start=1):
        if isinstance(row, dict):
            _check_unsafe_flags(row, path, label="source row")
            _check_no_secret_fields(row, path, label="source row")
            _check_no_nested_raw_identifier_fields(row, path, label="source row")
            yield idx, row


def _row_join_keys(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("canonical_group_id") or ""), str(row.get("order_key") or ""))


def _venue_id(row: dict[str, Any], fallback: str | None = None) -> str:
    return str(row.get("venue_id") or row.get("venue") or fallback or "").lower()


def _has_aster_role_fields(row: dict[str, Any]) -> bool:
    update = row.get("o") if isinstance(row.get("o"), dict) else row
    event_ok = row.get("e") == "ORDER_TRADE_UPDATE" or "m" in update or "maker_side" in update
    maker_flag = update.get("m", update.get("maker_side"))
    fill_qty = update.get("l", update.get("lastFilledQty"))
    return bool(event_ok and isinstance(maker_flag, bool) and _positive_float(fill_qty))


def _has_role_fields(row: dict[str, Any], venue: str) -> bool:
    if venue == "aster":
        return _has_aster_role_fields(row)
    if venue == "extended":
        return isinstance(row.get("isTaker") if "isTaker" in row else row.get("is_taker"), bool)
    if venue == "hyperliquid":
        return isinstance(row.get("crossed"), bool)
    if venue == "lighter":
        return (
            _safe_int(row.get("account_index")) is not None
            and isinstance(row.get("is_maker_ask") if "is_maker_ask" in row else row.get("isMakerAsk"), bool)
            and _safe_int(row.get("ask_account_id") or row.get("askAccountId")) is not None
            and _safe_int(row.get("bid_account_id") or row.get("bidAccountId")) is not None
        )
    if venue == "paradex":
        return str(row.get("liquidity") or "").upper() in {"MAKER", "TAKER"}
    return False


def _has_lighter_limit_fields(row: dict[str, Any]) -> bool:
    has_active = row.get("active_order_headroom_account") is not None and row.get("active_order_headroom_market") is not None
    has_sendtx = row.get("sendtx_per_minute_limit") is not None and row.get("sendtx_per_minute_remaining") is not None
    has_rest_or_weighted = (
        row.get("rest_requests_per_minute_limit") is not None
        and row.get("rest_requests_per_minute_remaining") is not None
    ) or (
        row.get("weighted_requests_per_minute_limit") is not None
        and row.get("weighted_requests_per_minute_remaining") is not None
    )
    return has_active and has_sendtx and has_rest_or_weighted and row.get("native_limit_event_time_status") in LIGHTER_LIMIT_ALIGNMENT_OK


def _pressure_state(row: dict[str, Any], path: Path, line_no: int) -> str:
    state = row.get("pressure_state")
    if state is None:
        return "pressure_complete" if _has_lighter_limit_fields(row) else "pressure_incomplete_or_unknown"
    if state == pressure_schema.PRESSURE_UNAVAILABLE:
        result = pressure_schema.validate_packet(row)
        if not result.accepted:
            joined = "; ".join(result.reject_reasons)
            raise ValueError(f"{path}:{line_no} invalid pressure_unavailable governance packet: {joined}")
        return pressure_schema.PRESSURE_UNAVAILABLE
    if state == pressure_schema.PRESSURE_COMPLETE:
        if not _has_lighter_limit_fields(row):
            raise ValueError(f"{path}:{line_no} pressure_complete row lacks complete explicit pressure dimensions")
        return pressure_schema.PRESSURE_COMPLETE
    if state == pressure_schema.PRESSURE_INCOMPLETE_OR_UNKNOWN:
        return pressure_schema.PRESSURE_INCOMPLETE_OR_UNKNOWN
    raise ValueError(f"{path}:{line_no} unsupported pressure_state {state!r}")


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"expected JSON object in {manifest_path}")
    _check_unsafe_flags(manifest, manifest_path, label="candidate manifest")
    _check_no_secret_fields(manifest, manifest_path, label="candidate manifest")
    if manifest.get("baseline_commit") not in {None, BASELINE_COMMIT}:
        raise ValueError(f"{manifest_path} baseline_commit mismatch")
    return manifest


def _phase51s_manifest_from_ready_entries(
    source_artifacts: list[dict[str, Any]],
    source_link_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = [
        {
            "source_id": item["source_id"],
            "venue_id": item["venue_id"],
            "path": item["resolved_path"],
        }
        for item in source_artifacts
        if item["status"] == "LOCAL_FILE_READY" and item.get("resolved_path")
    ]
    source_links = [
        {
            "source_link_id": item["source_link_id"],
            "path": item["resolved_path"],
        }
        for item in source_link_artifacts
        if item["status"] == "LOCAL_FILE_READY" and item.get("resolved_path")
    ]
    return {
        "manifest_version": 1,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_model_training": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "sources": sources,
        "source_links": source_links,
    }


def build_forward_capture_bundle_readiness(
    *,
    target_run: Path,
    candidate_manifest: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    target_run = _resolve_path(target_run)
    candidate_manifest = _resolve_path(candidate_manifest)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    target_summary, role_targets, limit_targets = _load_target_run(target_run)
    manifest = _load_manifest(candidate_manifest)
    role_by_group, role_by_order_key = _target_maps(role_targets)
    limit_by_group, limit_by_order_key = _target_maps(limit_targets)
    role_ready_target_ids: set[str] = set()
    limit_ready_target_ids: set[str] = set()
    labels: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    source_link_artifacts: list[dict[str, Any]] = []
    source_link_targets: dict[str, dict[str, Any]] = {}
    source_link_applied_row_count = 0
    lighter_pressure_state_counts: dict[str, int] = {}
    lighter_pressure_unavailable_source_count = 0
    seq = 0

    for index, source_link in enumerate(_manifest_paths(manifest.get("source_links"))):
        source_link_id = str(source_link.get("source_link_id") or f"source_link_{index}")
        path_text = str(source_link.get("path") or "")
        status, path = _source_status(path_text, candidate_manifest, entry_kind=f"source_links[{index}]")
        row_count = 0
        target_link_count = 0
        if path is not None and status == "LOCAL_FILE_READY":
            for line_no, row in _iter_jsonl(path):
                _check_unsafe_flags(row, path, label="source link")
                _check_no_secret_fields(row, path, label="source link")
                _check_no_nested_raw_identifier_fields(row, path, label="source link")
                unexpected = sorted(set(row) - SOURCE_LINK_ALLOWED_FIELDS)
                if unexpected:
                    raise ValueError(f"{path}:{line_no} source link has unsupported fields: {unexpected}")
                source_hashes = [
                    str(row.get(key) or "")
                    for key in SOURCE_LINK_HASH_FIELDS
                    if row.get(key)
                ]
                if not source_hashes:
                    raise ValueError(f"{path}:{line_no} source link missing source hash")
                group = str(row.get("canonical_group_id") or "")
                order_key = str(row.get("order_key") or "")
                if not group and not order_key:
                    raise ValueError(f"{path}:{line_no} source link missing canonical_group_id or order_key")
                role_target = _resolve_target_strict(
                    row,
                    role_by_group,
                    role_by_order_key,
                    path,
                    line_no,
                    target_label="native-role",
                )
                limit_target = _resolve_target_strict(
                    row,
                    limit_by_group,
                    limit_by_order_key,
                    path,
                    line_no,
                    target_label="native-limit",
                )
                if role_target is None and limit_target is None:
                    raise ValueError(f"{path}:{line_no} source link does not match Phase 5.1u targets")
                for source_hash in sorted(set(source_hashes)):
                    if source_hash in source_link_targets:
                        raise ValueError(f"{path}:{line_no} duplicate source link hash {source_hash}")
                    source_link_targets[source_hash] = {
                        "role_target": role_target,
                        "limit_target": limit_target,
                        "source_link_sha256": _stable_hash(row),
                    }
                row_count += 1
                target_link_count += 1
        source_link_artifacts.append(
            {
                "source_link_id": source_link_id,
                "path_hash": _stable_hash(path_text),
                "resolved_path": str(path) if path is not None and status == "LOCAL_FILE_READY" else None,
                "status": status,
                "row_count": row_count,
                "target_link_count": target_link_count,
                "sha256": _sha256_file(path) if path is not None and status == "LOCAL_FILE_READY" else None,
            }
        )

    for index, source in enumerate(_manifest_paths(manifest.get("sources"))):
        source_id = str(source.get("source_id") or f"source_{index}")
        venue_fallback = str(source.get("venue_id") or source.get("venue") or "").lower()
        path_text = str(source.get("path") or "")
        status, path = _source_status(path_text, candidate_manifest, entry_kind=f"source[{index}]")
        row_count = 0
        role_ready_count = 0
        limit_ready_count = 0
        if path is not None and status == "LOCAL_FILE_READY":
            for line_no, row in _iter_source_records(path):
                row_count += 1
                group, order_key = _row_join_keys(row)
                venue = _venue_id(row, venue_fallback)
                role_ready = False
                limit_ready = False
                lighter_pressure_state = "not_lighter"
                lighter_limit_governance_status = "NOT_APPLICABLE"
                role_target, role_join_status = _resolve_target_with_source_link(
                    row,
                    role_by_group,
                    role_by_order_key,
                    source_link_targets,
                    target_key="role_target",
                )
                if (
                    role_target is not None
                    and venue == str(role_target.get("venue_id") or "").lower()
                    and _has_role_fields(row, venue)
                ):
                    role_ready_target_ids.add(_target_id(role_target))
                    role_ready = True
                    role_ready_count += 1
                limit_target, limit_join_status = _resolve_target_with_source_link(
                    row,
                    limit_by_group,
                    limit_by_order_key,
                    source_link_targets,
                    target_key="limit_target",
                )
                if venue == "lighter":
                    lighter_pressure_state = _pressure_state(row, path, line_no)
                    lighter_pressure_state_counts[lighter_pressure_state] = (
                        lighter_pressure_state_counts.get(lighter_pressure_state, 0) + 1
                    )
                    if lighter_pressure_state == pressure_schema.PRESSURE_UNAVAILABLE:
                        lighter_pressure_unavailable_source_count += 1
                        lighter_limit_governance_status = PRESSURE_UNAVAILABLE_STATUS
                    elif lighter_pressure_state == pressure_schema.PRESSURE_COMPLETE:
                        lighter_limit_governance_status = "PRESSURE_COMPLETE"
                    else:
                        lighter_limit_governance_status = "PRESSURE_INCOMPLETE_OR_UNKNOWN"
                if (
                    limit_target is not None
                    and venue == "lighter"
                    and lighter_pressure_state == pressure_schema.PRESSURE_COMPLETE
                    and _has_lighter_limit_fields(row)
                ):
                    limit_ready_target_ids.add(_target_id(limit_target))
                    limit_ready = True
                    limit_ready_count += 1
                source_link_applied = role_join_status == "SOURCE_LINK_SIDECAR" or limit_join_status == "SOURCE_LINK_SIDECAR"
                if source_link_applied:
                    source_link_applied_row_count += 1
                label = _base_record(run_id, seq, timestamp_ns, "PHASE51V_SOURCE_ROW_READINESS_LABEL")
                seq += 1
                label.update(
                    {
                        "source_id": source_id,
                        "source_index": index,
                        "source_path_hash": _stable_hash(str(path)),
                        "source_line": line_no,
                        "source_row_sha256": _stable_hash(row),
                        "venue_id": venue or "unknown",
                        "canonical_group_id": group,
                        "order_key": order_key,
                        "role_target_join_status": role_join_status,
                        "lighter_limit_target_join_status": limit_join_status,
                        "source_link_applied": source_link_applied,
                        "source_row_readiness_status": "READY_FOR_TARGET" if role_ready or limit_ready else "NOT_TARGET_READY",
                        "role_target_ready": role_ready,
                        "lighter_limit_target_ready": limit_ready,
                        "lighter_limit_pressure_state": lighter_pressure_state,
                        "lighter_limit_governance_status": lighter_limit_governance_status,
                    }
                )
                labels.append(label)
        else:
            label = _base_record(run_id, seq, timestamp_ns, "PHASE51V_SOURCE_FILE_READINESS_LABEL")
            seq += 1
            label.update(
                {
                    "source_id": source_id,
                    "source_index": index,
                    "venue_id": venue_fallback or "unknown",
                    "source_path_hash": _stable_hash(path_text),
                    "source_file_readiness_status": status,
                    "role_target_ready": False,
                    "lighter_limit_target_ready": False,
                }
            )
            labels.append(label)
        source_artifacts.append(
            {
                "source_id": source_id,
                "venue_id": venue_fallback or "unknown",
                "path_hash": _stable_hash(path_text),
                "resolved_path": str(path) if path is not None and status == "LOCAL_FILE_READY" else None,
                "status": status,
                "row_count": row_count,
                "role_target_ready_row_count": role_ready_count,
                "lighter_limit_target_ready_row_count": limit_ready_count,
                "sha256": _sha256_file(path) if path is not None and status == "LOCAL_FILE_READY" else None,
            }
        )

    missing_role_targets = [
        row for row in role_targets if _target_id(row) not in role_ready_target_ids
    ]
    missing_limit_targets = [
        row for row in limit_targets if _target_id(row) not in limit_ready_target_ids
    ]
    limit_unavailable_count = len(missing_limit_targets) if lighter_pressure_unavailable_source_count else 0
    role_ready_count = len(role_targets) - len(missing_role_targets)
    limit_ready_count = len(limit_targets) - len(missing_limit_targets)
    bundle_ready = bool(role_targets or limit_targets) and not missing_role_targets and not missing_limit_targets
    source_owner_native_role_evidence_ready = bool(role_targets) and not missing_role_targets
    source_owner_native_role_evidence_status = (
        "READY"
        if source_owner_native_role_evidence_ready
        else "INCOMPLETE_OR_ABSENT"
    )
    gate_reason = (
        "phase51v_forward_capture_bundle_ready_for_phase51s_nonlive_hold"
        if bundle_ready
        else "phase51v_forward_capture_bundle_incomplete_nonlive_hold"
    )

    labels_path = out_dir / "capture_bundle_readiness_labels.jsonl"
    missing_role_path = out_dir / "missing_native_role_capture_targets.jsonl"
    missing_limit_path = out_dir / "missing_lighter_native_limit_capture_targets.jsonl"
    phase51s_manifest_path = out_dir / "phase51s_manifest.generated.json"
    summary_path = out_dir / "phase51v_forward_capture_bundle_readiness_summary.json"
    manifest_path = out_dir / "phase51v_manifest.json"
    _write_jsonl(labels_path, labels)
    _write_jsonl(missing_role_path, missing_role_targets)
    _write_jsonl(missing_limit_path, missing_limit_targets)
    generated_phase51s_manifest = _phase51s_manifest_from_ready_entries(source_artifacts, source_link_artifacts)
    _write_json(phase51s_manifest_path, generated_phase51s_manifest)

    summary = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
        "target_run": str(target_run),
        "target_manifest_summary_sha256": _sha256_file(target_run / "phase51u_forward_capture_target_manifest_summary.json"),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest),
        "native_role_capture_target_count": len(role_targets),
        "native_role_capture_target_ready_count": role_ready_count,
        "native_role_capture_target_missing_count": len(missing_role_targets),
        "source_owner_native_role_evidence_ready": source_owner_native_role_evidence_ready,
        "source_owner_native_role_evidence_status": source_owner_native_role_evidence_status,
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "lighter_native_limit_capture_target_ready_count": limit_ready_count,
        "lighter_native_limit_capture_target_missing_count": len(missing_limit_targets),
        "lighter_native_limit_pressure_unavailable_source_count": lighter_pressure_unavailable_source_count,
        "lighter_native_limit_pressure_unavailable_target_count": limit_unavailable_count,
        "lighter_native_limit_pressure_state_counts": dict(sorted(lighter_pressure_state_counts.items())),
        "lighter_native_limit_pressure_unavailable_status": (
            PRESSURE_UNAVAILABLE_STATUS if lighter_pressure_unavailable_source_count else "NOT_OBSERVED"
        ),
        "revised_pressure_unavailable_contract_observed": bool(lighter_pressure_unavailable_source_count),
        "revised_pressure_unavailable_contract_clears_blocker": False,
        "source_file_count": len(source_artifacts),
        "source_link_file_count": len(source_link_artifacts),
        "source_file_status_counts": _artifact_status_counts(source_artifacts),
        "source_link_file_status_counts": _artifact_status_counts(source_link_artifacts),
        "source_row_readiness_status_counts": _status_counts(labels, "source_row_readiness_status"),
        "source_link_hash_count": len(source_link_targets),
        "source_link_applied_row_count": source_link_applied_row_count,
        "generated_phase51s_manifest_path": str(phase51s_manifest_path),
        "generated_phase51s_source_count": len(generated_phase51s_manifest["sources"]),
        "generated_phase51s_source_link_count": len(generated_phase51s_manifest["source_links"]),
        "generated_phase51s_manifest_ready": bundle_ready,
        "source_artifacts": source_artifacts,
        "source_link_artifacts": source_link_artifacts,
        "downstream_chain_ready": bundle_ready,
        "downstream_next_command": (
            "python3 tools/phase51s_local_native_source_acquisition.py --manifest <candidate_manifest>"
            if bundle_ready
            else None
        ),
        "target_summary_gate_status": target_summary.get("gate_status"),
        "clears_phase51_blockers": False,
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
    _write_json(summary_path, summary)

    artifacts = [labels_path, missing_role_path, missing_limit_path, phase51s_manifest_path, summary_path]
    manifest_out = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "artifacts": _artifact_infos(out_dir, artifacts),
    }
    _write_json(manifest_path, manifest_out)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51v_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_forward_capture_bundle_readiness(
            target_run=args.target_run,
            candidate_manifest=args.candidate_manifest,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51v_forward_capture_bundle_readiness: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51v_forward_capture_bundle_readiness: wrote {out_dir}")
    print("phase51v_forward_capture_bundle_readiness: status HOLD (bundle readiness only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
