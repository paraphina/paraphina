#!/usr/bin/env python3
"""Phase 5.1al forward-refresh capture gate.

This HOLD-only gate materializes a fresh forward-refresh target pack from
already-sanitized, directly target-linkable native source rows. It emits a
Phase 5.1u-compatible target run, a strict Phase 5.1ae/5.1v-compatible
candidate manifest, and a minimal request pack so Phase 5.1ak can validate the
forward-refresh pack without composing current-pack artifacts.

It performs no network access, reads no env files, places no orders, cancels no
orders, does not call sendTx/sendTxBatch, and never infers native role, pressure
state, or source links from time/price/size/proximity.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51al_forward_refresh_capture_gate"

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

SECRET_FIELD_FRAGMENTS = {
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "session_token",
    "signing_key",
    "token",
}

RAW_IDENTIFIER_FIELDS = {
    "ask_client_id",
    "ask_id",
    "bid_client_id",
    "bid_id",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "decision_id",
    "fill_id",
    "fillId",
    "i",
    "id",
    "oid",
    "order_id",
    "orderId",
    "raw_client_order_id",
    "raw_order_id",
    "tid",
    "trade_id",
    "tradeId",
    "tx_hash",
    "txHash",
    "venue_order_id",
}

VENUES = {"aster", "extended", "hyperliquid", "lighter", "paradex"}
TARGET_TYPES = {"native_role", "lighter_native_limit"}
LIGHTER_LIMIT_ALIGNMENT_OK = {
    "EVENT_TIME_ALIGNED",
    "SNAPSHOT_AT_DECISION_TIME",
    "OBSERVED_AT_DECISION_TIME",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _field_looks_secret(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if "nonsecret" in normalized:
        return False
    if "authorization" in normalized and normalized.startswith(("no_", "not_")):
        return False
    return any(fragment in normalized for fragment in SECRET_FIELD_FRAGMENTS)


def _check_unsafe_flags(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_no_secret_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_no_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = sorted(RAW_IDENTIFIER_FIELDS & set(obj))
        if raw_fields:
            raise ValueError(f"{path} {label} leaked raw identifier fields: {raw_fields}")


def _check_safety(record: Any, path: Path, *, label: str) -> None:
    _check_unsafe_flags(record, path, label=label)
    _check_no_secret_fields(record, path, label=label)
    _check_no_raw_identifier_fields(record, path, label=label)


def _path_text_is_unsafe(path_text: str) -> bool:
    return "://" in path_text or path_text.lower().startswith(("http:", "https:", "s3:", "gs:"))


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink path is prohibited: {candidate}")


def _check_existing_jsonl(path: Path, *, label: str) -> Path:
    raw = str(path)
    if _path_text_is_unsafe(raw):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as Phase 5.1al {label} inputs")
    _check_no_symlink(resolved)
    if not resolved.exists():
        raise ValueError(f"{label} path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} path is not a file: {resolved}")
    if resolved.suffix != ".jsonl":
        raise ValueError(f"{label} path must be .jsonl: {resolved}")
    return resolved


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            _check_safety(record, path, label=f"input row {line_no}")
            yield line_no, record


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_safety(data, path, label="output")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            _check_safety(record, path, label="output")
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


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _venue_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        venue = str(record.get("venue_id") or "unknown")
        counts[venue] = counts.get(venue, 0) + 1
    return dict(sorted(counts.items()))


def _base_record(run_id: str, seq: int, timestamp_ns: int, label_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": label_type,
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "timestamp_utc": _timestamp_ns_to_utc(timestamp_ns + seq),
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


def _target_type(row: dict[str, Any], line_no: int) -> str:
    target_type = str(row.get("target_type") or row.get("target_kind") or "").strip().lower()
    if target_type not in TARGET_TYPES:
        raise ValueError(f"input row {line_no} target_type must be one of {sorted(TARGET_TYPES)}")
    return target_type


def _venue(row: dict[str, Any], line_no: int) -> str:
    venue = str(row.get("venue_id") or row.get("venue") or "").strip().lower()
    if venue not in VENUES:
        raise ValueError(f"input row {line_no} venue_id must be one of {sorted(VENUES)}")
    return venue


def _join_keys(row: dict[str, Any], line_no: int) -> tuple[str, str]:
    canonical_group_id = str(row.get("canonical_group_id") or "").strip()
    order_key = str(row.get("order_key") or "").strip()
    if not canonical_group_id and not order_key:
        raise ValueError(f"input row {line_no} requires canonical_group_id or order_key")
    return canonical_group_id, order_key


def _optional_text(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _role_payload(row: dict[str, Any], venue: str, line_no: int) -> dict[str, Any]:
    if venue == "aster":
        update = row.get("o") if isinstance(row.get("o"), dict) else row
        maker_flag = update.get("m", update.get("maker_side"))
        fill_qty = update.get("l", update.get("lastFilledQty"))
        if not isinstance(maker_flag, bool) or not _positive_float(fill_qty):
            raise ValueError(f"input row {line_no} missing Aster maker flag or positive fill quantity")
        return {
            "e": row.get("e") or "ORDER_TRADE_UPDATE",
            "m": maker_flag,
            "l": str(fill_qty),
        }
    if venue == "extended":
        value = row.get("isTaker") if "isTaker" in row else row.get("is_taker")
        if not isinstance(value, bool):
            raise ValueError(f"input row {line_no} missing Extended isTaker/is_taker boolean")
        return {"isTaker": value}
    if venue == "hyperliquid":
        value = row.get("crossed")
        if not isinstance(value, bool):
            raise ValueError(f"input row {line_no} missing Hyperliquid crossed boolean")
        return {"crossed": value}
    if venue == "lighter":
        account_index = _safe_int(row.get("account_index"))
        ask_account_id = _safe_int(row.get("ask_account_id") or row.get("askAccountId"))
        bid_account_id = _safe_int(row.get("bid_account_id") or row.get("bidAccountId"))
        is_maker_ask = row.get("is_maker_ask") if "is_maker_ask" in row else row.get("isMakerAsk")
        if (
            account_index is None
            or ask_account_id is None
            or bid_account_id is None
            or not isinstance(is_maker_ask, bool)
        ):
            raise ValueError(f"input row {line_no} missing Lighter account_index/is_maker_ask/account ids")
        return {
            "account_index": account_index,
            "is_maker_ask": is_maker_ask,
            "ask_account_id": ask_account_id,
            "bid_account_id": bid_account_id,
        }
    if venue == "paradex":
        liquidity = str(row.get("liquidity") or "").upper()
        if liquidity not in {"MAKER", "TAKER"}:
            raise ValueError(f"input row {line_no} missing Paradex liquidity MAKER/TAKER")
        return {"liquidity": liquidity}
    raise ValueError(f"input row {line_no} unsupported venue {venue}")


def _limit_payload(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    required = [
        "active_order_headroom_account",
        "active_order_headroom_market",
        "sendtx_per_minute_limit",
        "sendtx_per_minute_remaining",
    ]
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise ValueError(f"input row {line_no} missing Lighter native-limit fields: {missing}")

    has_rest = row.get("rest_requests_per_minute_limit") is not None and row.get(
        "rest_requests_per_minute_remaining"
    ) is not None
    has_weighted = row.get("weighted_requests_per_minute_limit") is not None and row.get(
        "weighted_requests_per_minute_remaining"
    ) is not None
    if not has_rest and not has_weighted:
        raise ValueError(f"input row {line_no} missing REST-or-weighted Lighter native-limit fields")

    alignment = str(row.get("native_limit_event_time_status") or "")
    if alignment not in LIGHTER_LIMIT_ALIGNMENT_OK:
        raise ValueError(f"input row {line_no} native_limit_event_time_status is not accepted")

    payload = {
        "active_order_headroom_account": row["active_order_headroom_account"],
        "active_order_headroom_market": row["active_order_headroom_market"],
        "sendtx_per_minute_limit": row["sendtx_per_minute_limit"],
        "sendtx_per_minute_remaining": row["sendtx_per_minute_remaining"],
        "native_limit_event_time_status": alignment,
    }
    if has_rest:
        payload["rest_requests_per_minute_limit"] = row["rest_requests_per_minute_limit"]
        payload["rest_requests_per_minute_remaining"] = row["rest_requests_per_minute_remaining"]
    if has_weighted:
        payload["weighted_requests_per_minute_limit"] = row["weighted_requests_per_minute_limit"]
        payload["weighted_requests_per_minute_remaining"] = row["weighted_requests_per_minute_remaining"]
    return payload


def _target_common(row: dict[str, Any], target_type: str, venue: str, line_no: int) -> dict[str, Any]:
    canonical_group_id, order_key = _join_keys(row, line_no)
    common = {
        "target_type": target_type,
        "venue_id": venue,
        "canonical_group_id": canonical_group_id,
        "order_key": order_key,
        "target_source": "PHASE51AL_FORWARD_REFRESH_CAPTURE_GATE",
        "target_join_status": "DIRECT_CANONICAL_KEY_REQUIRED",
        "source_link_inference_allowed": False,
        "time_price_size_inference_allowed": False,
    }
    for key in (
        "side",
        "price",
        "size",
        "symbol",
        "instrument_id",
        "market",
        "market_id",
        "order_id_hash",
        "client_order_id_hash",
        "decision_id_hash",
        "first_fill_time_ms",
        "last_fill_time_ms",
        "source_telemetry_sha256",
    ):
        if key in row and row[key] is not None:
            common[key] = row[key]
    return common


def _native_role_target(row: dict[str, Any], venue: str, line_no: int) -> dict[str, Any]:
    target = _target_common(row, "native_role", venue, line_no)
    target.update(
        {
            "target_reason": "forward_refresh_native_role_source_truth_required",
            "native_role_capture_status": "SOURCE_TRUTH_CAPTURED_IN_FORWARD_REFRESH",
            "native_role_missing_reason": None,
            "fill_count": row.get("fill_count", 1),
            "known_native_role_count": 1,
            "missing_native_role_count": 0,
            "required_native_role_source": {
                "aster": "ASTER_ORDER_TRADE_UPDATE_M",
                "extended": "EXTENDED_ISTAKER",
                "hyperliquid": "HYPERLIQUID_CROSSED",
                "lighter": "LIGHTER_TRADES_JSON",
                "paradex": "PARADEX_LIQUIDITY",
            }[venue],
        }
    )
    return target


def _lighter_limit_target(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    target = _target_common(row, "lighter_native_limit", "lighter", line_no)
    target.update(
        {
            "target_reason": "forward_refresh_lighter_event_time_pressure_required",
            "lighter_native_limit_capture_status": "SOURCE_TRUTH_CAPTURED_IN_FORWARD_REFRESH",
            "lighter_native_limit_missing_reason": None,
            "required_lighter_native_limit_fields": [
                "active_order_headroom_account",
                "active_order_headroom_market",
                "sendtx_per_minute_limit",
                "sendtx_per_minute_remaining",
                "rest_or_weighted_limit_remaining",
                "native_limit_event_time_status",
            ],
        }
    )
    return target


def _source_common(
    *,
    run_id: str,
    seq: int,
    timestamp_ns: int,
    row: dict[str, Any],
    target_type: str,
    venue: str,
    line_no: int,
    label_type: str,
) -> dict[str, Any]:
    canonical_group_id, order_key = _join_keys(row, line_no)
    source = _base_record(run_id, seq, timestamp_ns, label_type)
    source.update(
        {
            "target_type": target_type,
            "venue_id": venue,
            "canonical_group_id": canonical_group_id,
            "order_key": order_key,
            "source_record_sha256": _stable_hash(
                {
                    "input_line_no": line_no,
                    "target_type": target_type,
                    "venue_id": venue,
                    "canonical_group_id": canonical_group_id,
                    "order_key": order_key,
                    "sanitized_payload_hash": _stable_hash(row),
                }
            ),
            "source_record_origin": "PHASE51AL_SANITIZED_FORWARD_REFRESH_ROW",
            "source_link_status": "DIRECT_TARGET_LINKABLE",
            "source_link_inference_allowed": False,
            "time_price_size_inference_allowed": False,
        }
    )
    for key in (
        "order_id_hash",
        "client_order_id_hash",
        "decision_id_hash",
        "source_telemetry_sha256",
        "event_time_ms",
        "event_time_ns",
        "observed_at_ms",
        "observed_at_ns",
    ):
        if key in row and row[key] is not None:
            source[key] = row[key]
    return source


def _source_link_row(source_row: dict[str, Any]) -> dict[str, Any]:
    # Phase 5.1v intentionally accepts only source hashes, canonical join keys,
    # and false unsafe flags in source-link sidecars.
    return {
        "source_record_sha256": source_row["source_record_sha256"],
        "canonical_group_id": source_row.get("canonical_group_id") or "",
        "order_key": source_row.get("order_key") or "",
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }


def _load_forward_refresh_rows(input_jsonl: Path, run_id: str, timestamp_ns: int) -> dict[str, list[dict[str, Any]]]:
    role_targets: list[dict[str, Any]] = []
    limit_targets: list[dict[str, Any]] = []
    role_sources: list[dict[str, Any]] = []
    limit_sources: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str, str, str]] = set()

    source_seq = 0
    for line_no, row in _iter_jsonl(input_jsonl):
        target_type = _target_type(row, line_no)
        venue = _venue(row, line_no)
        canonical_group_id, order_key = _join_keys(row, line_no)
        target_key = (target_type, venue, canonical_group_id, order_key)
        if target_key in seen_targets:
            raise ValueError(f"input row {line_no} duplicates target {target_key}")
        seen_targets.add(target_key)

        if target_type == "native_role":
            role_payload = _role_payload(row, venue, line_no)
            target = _native_role_target(row, venue, line_no)
            source = _source_common(
                run_id=run_id,
                seq=source_seq,
                timestamp_ns=timestamp_ns,
                row=row,
                target_type=target_type,
                venue=venue,
                line_no=line_no,
                label_type="PHASE51AL_FORWARD_REFRESH_NATIVE_ROLE_SOURCE",
            )
            source.update(role_payload)
            role_targets.append({**_base_record(run_id, len(role_targets), timestamp_ns, "PHASE51AL_NATIVE_ROLE_TARGET"), **target})
            role_sources.append(source)
        else:
            if venue != "lighter":
                raise ValueError(f"input row {line_no} lighter_native_limit target must use venue_id=lighter")
            limit_payload = _limit_payload(row, line_no)
            target = _lighter_limit_target(row, line_no)
            source = _source_common(
                run_id=run_id,
                seq=source_seq,
                timestamp_ns=timestamp_ns,
                row=row,
                target_type=target_type,
                venue="lighter",
                line_no=line_no,
                label_type="PHASE51AL_LIGHTER_NATIVE_LIMIT_PRESSURE_SOURCE",
            )
            source.update(limit_payload)
            limit_targets.append(
                {**_base_record(run_id, len(limit_targets), timestamp_ns, "PHASE51AL_LIGHTER_NATIVE_LIMIT_TARGET"), **target}
            )
            limit_sources.append(source)

        label = _base_record(run_id, len(labels), timestamp_ns, "PHASE51AL_FORWARD_REFRESH_CAPTURE_LABEL")
        label.update(
            {
                "input_line_no": line_no,
                "target_type": target_type,
                "venue_id": venue,
                "canonical_group_id": canonical_group_id,
                "order_key": order_key,
                "source_record_sha256": source["source_record_sha256"],
                "capture_status": "ACCEPTED_FORWARD_REFRESH_SOURCE_TRUTH",
            }
        )
        labels.append(label)
        source_seq += 1

    if not role_targets and not limit_targets:
        raise ValueError("input-jsonl contains no forward-refresh rows")

    return {
        "role_targets": role_targets,
        "limit_targets": limit_targets,
        "role_sources": role_sources,
        "limit_sources": limit_sources,
        "labels": labels,
    }


def build_forward_refresh_capture_gate(
    *,
    input_jsonl: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    run_id = _check_run_id(run_id)
    input_jsonl = _check_existing_jsonl(input_jsonl, label="input-jsonl")
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    target_run = out_dir / "target_run"
    source_snapshots = out_dir / "source_snapshots"
    request_pack = out_dir / "phase51al_request_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _load_forward_refresh_rows(input_jsonl, run_id, timestamp_ns)
    role_targets = records["role_targets"]
    limit_targets = records["limit_targets"]
    role_sources = records["role_sources"]
    limit_sources = records["limit_sources"]
    labels = records["labels"]

    role_target_path = target_run / "native_role_capture_targets.jsonl"
    limit_target_path = target_run / "lighter_native_limit_capture_targets.jsonl"
    role_source_path = source_snapshots / "phase51al_forward_refresh_native_role_source.jsonl"
    limit_source_path = source_snapshots / "phase51al_lighter_forward_native_limit_pressure.jsonl"
    source_link_path = out_dir / "source_links.proposed.sanitized.jsonl"
    candidate_manifest_path = out_dir / "candidate_manifest.forward_refresh.json"
    labels_path = out_dir / "phase51al_forward_refresh_capture_labels.jsonl"
    summary_path = out_dir / "phase51al_forward_refresh_capture_summary.json"
    target_summary_path = target_run / "phase51u_forward_capture_target_manifest_summary.json"
    target_manifest_path = target_run / "phase51u_manifest.json"
    bundle_template_path = target_run / "capture_bundle_manifest_template.json"
    request_pack_manifest_path = request_pack / "manifest.json"
    request_targets_path = request_pack / "source_link_request_targets.jsonl"
    request_sources_path = request_pack / "source_link_request_sources.jsonl"
    manifest_path = out_dir / "manifest.json"
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"

    _write_jsonl(role_target_path, role_targets)
    _write_jsonl(limit_target_path, limit_targets)
    _write_jsonl(role_source_path, role_sources)
    _write_jsonl(limit_source_path, limit_sources)
    source_links = [_source_link_row(source) for source in role_sources + limit_sources]
    _write_jsonl(source_link_path, source_links)
    _write_jsonl(labels_path, labels)

    target_summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51al_forward_refresh_target_pack_materialized_nonlive_hold",
        "native_role_capture_target_count": len(role_targets),
        "native_role_capture_target_counts_by_venue": _venue_counts(role_targets),
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "source_truth_capture_status": "FORWARD_REFRESH_DIRECT_TARGET_LINKS_ONLY",
        "source_link_inference_allowed": False,
        "time_price_size_inference_allowed": False,
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
    _write_json(target_summary_path, target_summary)

    candidate_manifest = {
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
        "sources": [
            {
                "source_id": "phase51al_forward_refresh_native_role_source",
                "venue_id": "all5",
                "path": str(role_source_path),
            },
            {
                "source_id": "phase51al_lighter_forward_native_limit_pressure",
                "venue_id": "lighter",
                "path": str(limit_source_path),
            },
        ],
        "source_links": [
            {
                "source_link_id": "phase51al_forward_refresh_direct_source_links",
                "path": str(source_link_path),
            }
        ],
    }
    _write_json(candidate_manifest_path, candidate_manifest)

    bundle_template = {
        "manifest_version": 1,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "candidate_manifest_path": str(candidate_manifest_path),
        "phase51v_command": (
            "python3 tools/phase51v_forward_capture_bundle_readiness.py "
            f"--target-run {target_run} "
            f"--candidate-manifest {candidate_manifest_path}"
        ),
        "phase51ak_command": (
            "python3 tools/phase51ak_blocker_resolution_runner.py "
            f"--target-run {target_run} "
            f"--request-pack {request_pack} "
            "--no-default-current-manifest "
            f"--candidate-manifest {candidate_manifest_path} "
            "--target-pack-mode forward-refresh"
        ),
        "no_live_flag": True,
        "approved_for_live": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }
    _write_json(bundle_template_path, bundle_template)
    _write_json(target_manifest_path, {**target_summary, "artifacts": _artifact_infos(target_run, [role_target_path, limit_target_path, target_summary_path, bundle_template_path])})

    request_targets = []
    for seq, target in enumerate(role_targets + limit_targets):
        request_target = _base_record(run_id, seq, timestamp_ns, "PHASE51AL_SOURCE_LINK_REQUEST_TARGET")
        request_target.update(
            {
                "target_type": target["target_type"],
                "venue_id": target["venue_id"],
                "canonical_group_id": target.get("canonical_group_id") or "",
                "order_key": target.get("order_key") or "",
                "source_link_status": "ALREADY_DIRECT_TARGET_LINKABLE",
                "source_link_inference_allowed": False,
                "time_price_size_inference_allowed": False,
            }
        )
        request_targets.append(request_target)
    _write_json(request_pack_manifest_path, {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "source_link_request_status": "FORWARD_REFRESH_DIRECT_TARGET_LINKABLE",
        "no_live_flag": True,
        "approved_for_live": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    })
    _write_jsonl(request_targets_path, request_targets)
    _write_jsonl(request_sources_path, [])

    artifact_paths = [
        role_target_path,
        limit_target_path,
        role_source_path,
        limit_source_path,
        source_link_path,
        candidate_manifest_path,
        labels_path,
        target_summary_path,
        target_manifest_path,
        bundle_template_path,
        request_pack_manifest_path,
        request_targets_path,
        request_sources_path,
    ]
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51al_forward_refresh_capture_gate_nonlive_hold",
        "input_jsonl_sha256": _sha256_file(input_jsonl),
        "target_run": str(target_run),
        "request_pack": str(request_pack),
        "candidate_manifest_path": str(candidate_manifest_path),
        "native_role_capture_target_count": len(role_targets),
        "native_role_capture_target_counts_by_venue": _venue_counts(role_targets),
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "source_row_count": len(role_sources) + len(limit_sources),
        "source_row_counts_by_label": _status_counts(role_sources + limit_sources, "label_type"),
        "source_link_count": len(source_links),
        "phase51ak_forward_refresh_command": bundle_template["phase51ak_command"],
        "promotion_boundary": "Phase 5.1ak/5.1v readiness only; no live or economic claim",
        "clears_phase51_blockers": False,
        "next_required_action": "run_phase51ak_with_target_pack_mode_forward_refresh",
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
    artifact_paths.append(summary_path)

    artifact_index = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "artifacts": _artifact_infos(out_dir, artifact_paths),
        "no_live_flag": True,
        "approved_for_live": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }
    _write_json(artifact_index_path, artifact_index)
    artifact_paths.append(artifact_index_path)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "target_run": str(target_run),
            "candidate_manifest_path": str(candidate_manifest_path),
            "request_pack": str(request_pack),
            "artifacts": _artifact_infos(out_dir, artifact_paths),
            "clears_phase51_blockers": False,
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
        },
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AL-FORWARD-REFRESH-CAPTURE-GATE-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int)
    args = parser.parse_args()

    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_forward_refresh_capture_gate(
            input_jsonl=args.input_jsonl,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail closed with a concise error
        print(f"phase51al_forward_refresh_capture_gate: ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"phase51al_forward_refresh_capture_gate: wrote {out_dir}")
    print("phase51al_forward_refresh_capture_gate: status HOLD (non-live forward-refresh capture gate only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
