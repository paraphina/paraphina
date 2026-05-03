#!/usr/bin/env python3
"""Stage local Phase 5.1 native source snapshots for Phase 5.1r.

This HOLD-only tool is a safety preflight in front of Phase 5.1r. It accepts
an explicit local manifest, rejects network/env/symlink/secret-shaped sources,
redacts raw venue identifiers, and emits one sanitized JSONL source file that
can be passed to ``tools/phase51r_forward_native_source_acquisition.py``.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51s_local_native_source_acquisition"
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


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    raw_fields = RAW_IDENTIFIER_FIELDS & set(record)
    if raw_fields:
        raise ValueError(f"{path} output leaked raw identifier fields: {sorted(raw_fields)}")


def _check_no_nested_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = NESTED_RAW_IDENTIFIER_FIELDS & set(obj)
        if raw_fields:
            raise ValueError(f"{path} {label} leaked raw identifier fields: {sorted(raw_fields)}")


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


def _path_text_is_unsafe(path_text: str) -> bool:
    lower = path_text.lower()
    return lower.startswith(("http://", "https://")) or "://" in lower


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink source path is prohibited: {candidate}")


def _source_path(entry: dict[str, Any], manifest_path: Path, index: int, *, entry_kind: str = "source") -> Path:
    raw = entry.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{manifest_path}: {entry_kind}[{index}] missing local path")
    if _path_text_is_unsafe(raw):
        raise ValueError(f"{manifest_path}: network {entry_kind} paths are prohibited")
    path = _resolve_path(Path(raw))
    if _is_env_path(path):
        raise ValueError(f"{manifest_path}: env files are prohibited as native {entry_kind} input")
    _check_no_symlink(path)
    if not path.exists():
        raise ValueError(f"{manifest_path}: {entry_kind} path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{manifest_path}: {entry_kind} path must be a file: {path}")
    if path.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"{manifest_path}: {entry_kind} path must be .json or .jsonl: {path}")
    return path


def _top_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "account_index",
            "canonical_group_id",
            "market",
            "market_id",
            "market_symbol",
            "native_limit_event_time_status",
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
            _check_no_secret_fields(row, path, label="source row")
            for item in _payload_records(row):
                yield line_no, item
        return
    payload = _load_json(path)
    _check_unsafe_flags(payload, path, label="source payload")
    _check_no_secret_fields(payload, path, label="source payload")
    for idx, row in enumerate(_payload_records(payload), start=1):
        if isinstance(row, dict):
            _check_unsafe_flags(row, path, label="source row")
            _check_no_secret_fields(row, path, label="source row")
            yield idx, row


def _redact_value(value: Any, stripped: list[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            key_str = str(key)
            if key_str in NESTED_RAW_IDENTIFIER_FIELDS:
                stripped.append(key_str)
                continue
            if _field_looks_secret(key_str):
                raise ValueError(f"secret-shaped source field {key_str!r}")
            redacted_child = _redact_value(child, stripped)
            if redacted_child is not None:
                out[key_str] = redacted_child
        return out
    if isinstance(value, list):
        return [_redact_value(child, stripped) for child in value]
    return value


def _redact_source_row(row: dict[str, Any], venue_id: str | None) -> tuple[dict[str, Any], int]:
    stripped: list[str] = []
    redacted = _redact_value(row, stripped)
    if not isinstance(redacted, dict):
        return {}, len(stripped)
    if venue_id and not redacted.get("venue_id") and not redacted.get("venue"):
        redacted["venue_id"] = venue_id
    redacted["phase51s_source_record_sha256"] = _stable_hash(row)
    redacted["phase51s_raw_identifier_fields_stripped_count"] = len(stripped)
    redacted.setdefault("approved_for_model_training", False)
    redacted.setdefault("approved_for_live", False)
    redacted.setdefault("approved_for_canary", False)
    redacted.setdefault("approved_for_capital_escalation", False)
    redacted.setdefault("admissible_for_financial_claim", False)
    redacted.setdefault("admissible_for_ev_admission", False)
    redacted.setdefault("live_orders_allowed", False)
    redacted.setdefault("capital_change_allowed", False)
    redacted.setdefault("risk_limit_relaxation_allowed", False)
    _check_output_safe(redacted, Path("phase51s_redacted_source_row"))
    return redacted, len(stripped)


def _source_link_hashes(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in sorted(SOURCE_LINK_HASH_FIELDS):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _require_source_link_string(row: dict[str, Any], path: Path, line_no: int, key: str) -> None:
    value = row.get(key)
    if value is None or value == "":
        return
    if not isinstance(value, str):
        raise ValueError(f"{path}:{line_no} source link field {key} must be a string")


def _check_source_link_safe(row: dict[str, Any], path: Path, line_no: int) -> None:
    _check_unsafe_flags(row, path, label="source link")
    _check_no_secret_fields(row, path, label="source link")
    _check_no_nested_raw_identifier_fields(row, path, label="source link")
    unexpected = sorted(set(row) - SOURCE_LINK_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"{path}:{line_no} source link has unsupported fields: {unexpected}")
    for key in sorted(SOURCE_LINK_HASH_FIELDS | {"canonical_group_id", "order_key"}):
        _require_source_link_string(row, path, line_no, key)
    if not _source_link_hashes(row):
        raise ValueError(f"{path}:{line_no} source link missing source hash")
    if not (row.get("canonical_group_id") or row.get("order_key")):
        raise ValueError(f"{path}:{line_no} source link missing canonical_group_id or order_key")


def _iter_source_link_records(path: Path):
    if path.suffix == ".jsonl":
        for line_no, row in _iter_jsonl(path):
            _check_source_link_safe(row, path, line_no)
            yield line_no, row
        return
    payload = _load_json(path)
    _check_unsafe_flags(payload, path, label="source link payload")
    _check_no_secret_fields(payload, path, label="source link payload")
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("source_links"), list):
        rows = payload["source_links"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError(f"expected JSON object at {path}:1")
    for line_no, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        _check_source_link_safe(row, path, line_no)
        yield line_no, row


def _stage_source_link_row(row: dict[str, Any]) -> dict[str, Any]:
    staged = {
        key: row[key]
        for key in sorted(SOURCE_LINK_ALLOWED_FIELDS)
        if key in row
    }
    for flag in UNSAFE_TRUE_FLAGS:
        staged.setdefault(flag, False)
    _check_source_link_safe(staged, Path("phase51s_source_link_sidecar"), 0)
    return staged


def _has_join_key(row: dict[str, Any]) -> bool:
    return bool(row.get("canonical_group_id") or row.get("order_key"))


def _has_lighter_complete_limit_fields(row: dict[str, Any]) -> bool:
    if str(row.get("venue_id") or row.get("venue") or "").lower() != "lighter":
        return False
    has_active = row.get("active_order_headroom_account") is not None and row.get("active_order_headroom_market") is not None
    has_sendtx = row.get("sendtx_per_minute_limit") is not None and row.get("sendtx_per_minute_remaining") is not None
    has_rest_or_weighted = (
        row.get("rest_requests_per_minute_limit") is not None
        and row.get("rest_requests_per_minute_remaining") is not None
    ) or (
        row.get("weighted_requests_per_minute_limit") is not None
        and row.get("weighted_requests_per_minute_remaining") is not None
    )
    return has_active and has_sendtx and has_rest_or_weighted and row.get("native_limit_event_time_status") in {
        "EVENT_TIME_ALIGNED",
        "SNAPSHOT_AT_DECISION_TIME",
        "OBSERVED_AT_DECISION_TIME",
    }


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_local_native_source_acquisition(
    *,
    manifest: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    manifest = _resolve_path(manifest)
    output_root = _resolve_path(output_root)
    _check_no_symlink(manifest)
    if _is_env_path(manifest):
        raise ValueError("env files are prohibited as Phase 5.1s manifests")
    manifest_payload = _load_json(manifest)
    _check_unsafe_flags(manifest_payload, manifest, label="manifest")
    _check_no_secret_fields(manifest_payload, manifest, label="manifest")
    if manifest_payload.get("baseline_commit") not in {None, BASELINE_COMMIT}:
        raise ValueError(f"{manifest} baseline_commit mismatch")
    sources = manifest_payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{manifest} must contain a sources list")
    source_links = manifest_payload.get("source_links", [])
    if source_links is None:
        source_links = []
    if not isinstance(source_links, list):
        raise ValueError(f"{manifest} source_links must be a list when present")

    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    staged_records: list[dict[str, Any]] = []
    staged_source_links: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_link_labels: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    source_link_artifacts: list[dict[str, Any]] = []
    seq = 0
    source_link_seq = 0
    source_row_count = 0
    source_link_row_count = 0
    join_key_row_count = 0
    stripped_raw_identifier_field_count = 0
    complete_lighter_limit_row_count = 0
    source_link_hashes_seen: dict[str, dict[str, Any]] = {}

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"{manifest}: source[{index}] must be an object")
        _check_unsafe_flags(source, manifest, label=f"source[{index}]")
        _check_no_secret_fields(source, manifest, label=f"source[{index}]")
        path = _source_path(source, manifest, index)
        venue_id = str(source.get("venue_id") or source.get("venue") or "").lower() or None
        row_count = 0
        staged_count = 0
        stripped_count = 0
        for line_no, row in _iter_source_records(path):
            source_row_count += 1
            row_count += 1
            try:
                redacted, stripped = _redact_source_row(row, venue_id)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} {exc}") from exc
            stripped_raw_identifier_field_count += stripped
            stripped_count += stripped
            has_join_key = _has_join_key(redacted)
            has_complete_limit = _has_lighter_complete_limit_fields(redacted)
            if has_join_key:
                join_key_row_count += 1
            if has_complete_limit:
                complete_lighter_limit_row_count += 1
            staged_records.append(redacted)
            staged_count += 1

            status = "STAGED_LOCAL_SOURCE_ROW"
            if not has_join_key:
                status = "STAGED_LOCAL_SOURCE_ROW_WITHOUT_JOIN_KEY"
            label = _base_record(run_id, seq, timestamp_ns, "PHASE51S_LOCAL_SOURCE_LABEL")
            seq += 1
            label.update(
                {
                    "source_index": index,
                    "source_path_hash": _stable_hash(str(path)),
                    "source_line": line_no,
                    "source_record_sha256": _stable_hash(row),
                    "redacted_source_record_sha256": _stable_hash(redacted),
                    "venue_id": redacted.get("venue_id") or redacted.get("venue") or "unknown",
                    "local_source_stage_status": status,
                    "join_key_present": has_join_key,
                    "lighter_complete_native_limit_fields_present": has_complete_limit,
                    "raw_identifier_fields_stripped_count": stripped,
                }
            )
            labels.append(label)
        source_artifacts.append(
            {
                "path_hash": _stable_hash(str(path)),
                "sha256": _sha256_file(path),
                "row_count": row_count,
                "staged_count": staged_count,
                "raw_identifier_fields_stripped_count": stripped_count,
            }
        )

    for index, source_link in enumerate(source_links):
        if not isinstance(source_link, dict):
            raise ValueError(f"{manifest}: source_links[{index}] must be an object")
        _check_unsafe_flags(source_link, manifest, label=f"source_links[{index}]")
        _check_no_secret_fields(source_link, manifest, label=f"source_links[{index}]")
        path = _source_path(source_link, manifest, index, entry_kind="source_links")
        row_count = 0
        hash_count = 0
        for line_no, row in _iter_source_link_records(path):
            source_link_row_count += 1
            row_count += 1
            staged = _stage_source_link_row(row)
            row_hashes = _source_link_hashes(staged)
            for source_hash in row_hashes:
                existing = source_link_hashes_seen.get(source_hash)
                if existing is not None:
                    raise ValueError(f"{path}:{line_no} duplicate source link hash {source_hash}")
                source_link_hashes_seen[source_hash] = {
                    "canonical_group_id": staged.get("canonical_group_id") or "",
                    "order_key": staged.get("order_key") or "",
                }
            hash_count += len(row_hashes)
            staged_source_links.append(staged)

            label = _base_record(run_id, source_link_seq, timestamp_ns, "PHASE51S_SOURCE_LINK_LABEL")
            source_link_seq += 1
            label.update(
                {
                    "source_link_index": index,
                    "source_path_hash": _stable_hash(str(path)),
                    "source_line": line_no,
                    "source_link_sha256": _stable_hash(staged),
                    "source_link_hash_count": len(row_hashes),
                    "canonical_group_present": bool(staged.get("canonical_group_id")),
                    "order_key_present": bool(staged.get("order_key")),
                    "local_source_link_stage_status": "STAGED_LOCAL_SOURCE_LINK_ROW",
                }
            )
            source_link_labels.append(label)
        source_link_artifacts.append(
            {
                "path_hash": _stable_hash(str(path)),
                "sha256": _sha256_file(path),
                "row_count": row_count,
                "staged_count": row_count,
                "source_link_hash_count": hash_count,
            }
        )

    source_path = out_dir / "local_native_source.jsonl"
    source_link_path = out_dir / "local_source_link_sidecar.jsonl"
    labels_path = out_dir / "local_source_labels.jsonl"
    source_link_labels_path = out_dir / "local_source_link_labels.jsonl"
    summary_path = out_dir / "phase51s_local_native_source_acquisition_summary.json"
    manifest_path = out_dir / "phase51s_manifest.json"

    _write_jsonl(source_path, staged_records)
    _write_jsonl(source_link_path, staged_source_links)
    _write_jsonl(labels_path, labels)
    _write_jsonl(source_link_labels_path, source_link_labels)

    gate_reason = (
        "phase51s_local_native_source_acquisition_complete_nonlive_hold"
        if staged_records
        else "phase51s_local_native_source_acquisition_incomplete_source_links_only"
        if staged_source_links
        else "phase51s_local_native_source_acquisition_incomplete"
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
        "manifest_path_hash": _stable_hash(str(manifest)),
        "manifest_sha256": _sha256_file(manifest),
        "source_file_count": len(sources),
        "source_row_count": source_row_count,
        "staged_source_row_count": len(staged_records),
        "source_link_file_count": len(source_links),
        "source_link_row_count": source_link_row_count,
        "staged_source_link_row_count": len(staged_source_links),
        "source_link_hash_count": len(source_link_hashes_seen),
        "join_key_source_row_count": join_key_row_count,
        "source_row_without_join_key_count": len(staged_records) - join_key_row_count,
        "complete_lighter_native_limit_source_row_count": complete_lighter_limit_row_count,
        "raw_identifier_fields_stripped_count": stripped_raw_identifier_field_count,
        "local_source_stage_status_counts": _status_counts(labels, "local_source_stage_status"),
        "local_source_link_stage_status_counts": _status_counts(source_link_labels, "local_source_link_stage_status"),
        "source_artifacts": source_artifacts,
        "source_link_artifacts": source_link_artifacts,
        "downstream_tool": "tools/phase51r_forward_native_source_acquisition.py",
        "downstream_argument": "--source-json local_native_source.jsonl",
        "downstream_source_link_argument": "--source-link-jsonl local_source_link_sidecar.jsonl",
        "clears_phase51_blockers": False,
    }
    _write_json(summary_path, summary)

    artifacts = [source_path, source_link_path, labels_path, source_link_labels_path, summary_path]
    manifest_out = {
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51s_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_local_native_source_acquisition(
            manifest=args.manifest,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51s_local_native_source_acquisition: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51s_local_native_source_acquisition: wrote {out_dir}")
    print("phase51s_local_native_source_acquisition: status HOLD (local non-live source staging only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
