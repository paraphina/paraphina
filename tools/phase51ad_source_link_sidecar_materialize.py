#!/usr/bin/env python3
"""Materialize a validated redacted source-link sidecar for a request pack.

This HOLD-only tool consumes a Phase 5.1z source-link request pack and a
reviewer/collector-provided redacted mapping file. It validates that every
mapping uses only request-pack source hashes and Phase 5.1u target join keys,
then emits a Phase 5.1v-compatible `source_links.sanitized.jsonl` and candidate
manifest.

It does not infer links, read secrets, emit raw identifiers, place orders,
enable live/canary behavior, escalate capital, or relax risk.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ad_source_link_sidecar_materialize"

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
    "token",
}

RAW_IDENTIFIER_FIELDS = {
    "ask_client_id",
    "ask_client_id_str",
    "ask_id",
    "ask_id_str",
    "bid_client_id",
    "bid_client_id_str",
    "bid_id",
    "bid_id_str",
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
    "order_id_str",
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

SOURCE_HASH_FIELDS = {
    "phase51s_source_record_sha256",
    "redacted_source_record_sha256",
    "source_record_sha256",
}

MAPPING_ALLOWED_FIELDS = (
    SOURCE_HASH_FIELDS
    | {"canonical_group_id", "order_key", "venue_id", "no_live_flag"}
    | UNSAFE_TRUE_FLAGS
)

OUTPUT_ALLOWED_FIELDS = SOURCE_HASH_FIELDS | {"canonical_group_id", "order_key"}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _is_uri_like(value: str) -> bool:
    return "://" in value or value.startswith(("http:", "https:", "s3:", "gs:"))


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink path is prohibited: {candidate}")


def _check_local_path(path: Path, *, label: str) -> Path:
    raw = str(path)
    if _is_uri_like(raw):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as Phase 5.1ad {label} inputs")
    _check_no_symlink(resolved)
    return resolved


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]], *, sidecar_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if sidecar_output:
                _check_output_safe(record, path)
            else:
                _check_unsafe_flags(record, path, label="output")
                _check_no_secret_fields(record, path, label="output")
                _check_no_raw_identifier_fields(record, path, label="output")
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


def _iter_mapping_records(path: Path):
    if path.suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    if path.suffix != ".json":
        raise ValueError(f"mapping path must be .json or .jsonl: {path}")
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("source_links"), list):
        records = payload["source_links"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError(f"expected JSON object or array in {path}")
    for line_no, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"expected mapping object at {path}:{line_no}")
        yield line_no, record


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


def _check_no_secret_fields(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_no_raw_identifier_fields(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = sorted(str(key) for key in obj if str(key) in RAW_IDENTIFIER_FIELDS)
        if raw_fields:
            raise ValueError(f"{path} leaked raw identifier {label} fields: {raw_fields}")


def _check_unsafe_flags(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        if "no_live_flag" in obj and obj.get("no_live_flag") is not True:
            raise ValueError(f"{path} has unsafe {label} flag no_live_flag={obj.get('no_live_flag')!r}")
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    unexpected = sorted(set(record) - OUTPUT_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"{path} output sidecar has unsupported fields: {unexpected}")
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


def _source_hashes(row: dict[str, Any], path: Path, line_no: int) -> list[str]:
    out: list[str] = []
    for field in sorted(SOURCE_HASH_FIELDS):
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{path}:{line_no} source hash field {field} must be a string")
        if not _is_sha256_hex(value):
            raise ValueError(f"{path}:{line_no} source hash field {field} must be sha256 hex")
        out.append(value.lower())
    return sorted(set(out))


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
        "approved_for_financial_claim": False,
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
    }


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _load_request_sources(request_pack: Path) -> dict[str, dict[str, Any]]:
    path = request_pack / "source_link_request_sources.jsonl"
    if not path.exists():
        raise ValueError(f"missing request-pack file: {path}")
    sources: dict[str, dict[str, Any]] = {}
    for line_no, row in _iter_jsonl(path):
        _check_unsafe_flags(row, path, label="request source")
        _check_no_secret_fields(row, path, label="request source")
        _check_no_raw_identifier_fields(row, path, label="request source")
        source_hashes = _source_hashes(row, path, line_no)
        if len(source_hashes) != 1:
            raise ValueError(f"{path}:{line_no} request source must have exactly one source_record_sha256")
        source_hash = source_hashes[0]
        if source_hash in sources:
            raise ValueError(f"{path}:{line_no} duplicate request source hash {source_hash}")
        sources[source_hash] = dict(row)
    if not sources:
        raise ValueError("request pack has no source rows")
    return sources


def _target_id(row: dict[str, Any]) -> str:
    return str(row.get("canonical_group_id") or row.get("order_key") or "")


def _register_target(
    index: dict[str, dict[str, Any]],
    key: Any,
    row: dict[str, Any],
    path: Path,
    line_no: int,
) -> None:
    if not key:
        return
    key_text = str(key)
    existing = index.get(key_text)
    if existing is not None and _target_id(existing) != _target_id(row):
        raise ValueError(f"{path}:{line_no} duplicate target key maps to conflicting targets")
    index[key_text] = row


def _load_request_targets(request_pack: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    path = request_pack / "source_link_request_targets.jsonl"
    if not path.exists():
        raise ValueError(f"missing request-pack file: {path}")
    by_group: dict[str, dict[str, Any]] = {}
    by_order_key: dict[str, dict[str, Any]] = {}
    targets: list[dict[str, Any]] = []
    for line_no, row in _iter_jsonl(path):
        _check_unsafe_flags(row, path, label="request target")
        _check_no_secret_fields(row, path, label="request target")
        _check_no_raw_identifier_fields(row, path, label="request target")
        if not row.get("canonical_group_id") and not row.get("order_key"):
            raise ValueError(f"{path}:{line_no} request target missing canonical_group_id or order_key")
        targets.append(dict(row))
        _register_target(by_group, row.get("canonical_group_id"), row, path, line_no)
        _register_target(by_order_key, row.get("order_key"), row, path, line_no)
    if not targets:
        raise ValueError("request pack has no target rows")
    return by_group, by_order_key, targets


def _count_by_venue(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        venue = str(row.get("venue_id") or "unknown").lower()
        counts[venue] = counts.get(venue, 0) + 1
    return dict(sorted(counts.items()))


def _resolve_target(
    mapping: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
    path: Path,
    line_no: int,
) -> dict[str, Any]:
    group = str(mapping.get("canonical_group_id") or "")
    order_key = str(mapping.get("order_key") or "")
    if not group and not order_key:
        raise ValueError(f"{path}:{line_no} mapping missing canonical_group_id or order_key")
    group_target = by_group.get(group) if group else None
    order_target = by_order_key.get(order_key) if order_key else None
    if group and group_target is None:
        raise ValueError(f"{path}:{line_no} unknown canonical_group_id {group}")
    if order_key and order_target is None:
        raise ValueError(f"{path}:{line_no} unknown order_key {order_key}")
    if group_target is not None and order_target is not None and _target_id(group_target) != _target_id(order_target):
        raise ValueError(f"{path}:{line_no} canonical_group_id conflicts with order_key")
    target = group_target or order_target
    if target is None:
        raise ValueError(f"{path}:{line_no} mapping does not match request targets")
    return target


def _load_mapping_rows(
    *,
    mapping_path: Path,
    request_sources: dict[str, dict[str, Any]],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sidecars: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    seen_source_hashes: set[str] = set()
    for seq, (line_no, row) in enumerate(_iter_mapping_records(mapping_path)):
        _check_unsafe_flags(row, mapping_path, label="mapping")
        _check_no_secret_fields(row, mapping_path, label="mapping")
        _check_no_raw_identifier_fields(row, mapping_path, label="mapping")
        unexpected = sorted(set(row) - MAPPING_ALLOWED_FIELDS)
        if unexpected:
            raise ValueError(f"{mapping_path}:{line_no} unsupported mapping fields: {unexpected}")
        source_hashes = _source_hashes(row, mapping_path, line_no)
        if len(source_hashes) != 1:
            raise ValueError(f"{mapping_path}:{line_no} mapping must include exactly one source hash")
        source_hash = source_hashes[0]
        if source_hash in seen_source_hashes:
            raise ValueError(f"{mapping_path}:{line_no} duplicate mapping for source hash {source_hash}")
        source_row = request_sources.get(source_hash)
        if source_row is None:
            raise ValueError(f"{mapping_path}:{line_no} source hash not found in request pack: {source_hash}")
        target = _resolve_target(row, by_group, by_order_key, mapping_path, line_no)
        source_venue = str(source_row.get("venue_id") or "").lower()
        target_venue = str(target.get("venue_id") or "").lower()
        mapping_venue = str(row.get("venue_id") or source_venue).lower()
        if mapping_venue and mapping_venue != source_venue:
            raise ValueError(f"{mapping_path}:{line_no} mapping venue_id does not match request source")
        if source_venue and target_venue and source_venue != target_venue:
            raise ValueError(f"{mapping_path}:{line_no} request source venue does not match target venue")
        sidecar = {
            "source_record_sha256": source_hash,
            "canonical_group_id": target.get("canonical_group_id"),
        }
        if target.get("order_key"):
            sidecar["order_key"] = target.get("order_key")
        sidecars.append(sidecar)
        seen_source_hashes.add(source_hash)
        labels.append(
            {
                "source_record_sha256": source_hash,
                "canonical_group_id": target.get("canonical_group_id"),
                "order_key": target.get("order_key"),
                "venue_id": source_venue or target_venue or "unknown",
                "mapping_line": line_no,
                "source_link_materialize_status": "SOURCE_LINK_MATERIALIZED",
                "source_link_sha256": _stable_hash(sidecar),
                "source_row_sha256": _stable_hash(source_row),
                "target_row_sha256": _stable_hash(target),
                "mapping_row_sha256": _stable_hash(row),
                "label_seq": seq,
            }
        )
    if not sidecars:
        raise ValueError("mapping file has no materializable source-link rows")
    return sorted(sidecars, key=lambda item: item["source_record_sha256"]), labels


def _load_summary(request_pack: Path) -> dict[str, Any]:
    path = request_pack / "phase51z_source_link_request_pack_summary.json"
    if not path.exists():
        raise ValueError(f"missing request-pack summary: {path}")
    summary = _load_json(path)
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {path}")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{path} baseline_commit mismatch")
    _check_unsafe_flags(summary, path, label="request summary")
    _check_no_secret_fields(summary, path, label="request summary")
    return summary


def _materialized_candidate_manifest(
    request_pack: Path,
    summary: dict[str, Any],
    sidecar_path: Path,
) -> dict[str, Any]:
    manifest_path_text = summary.get("candidate_manifest_with_empty_sidecar")
    if manifest_path_text:
        manifest_path = _check_local_path(Path(str(manifest_path_text)), label="empty candidate manifest")
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"expected JSON object in {manifest_path}")
        _check_unsafe_flags(manifest, manifest_path, label="empty candidate manifest")
        _check_no_secret_fields(manifest, manifest_path, label="empty candidate manifest")
        _check_no_raw_identifier_fields(manifest, manifest_path, label="empty candidate manifest")
        for index, source in enumerate(manifest.get("sources") or []):
            if not isinstance(source, dict):
                raise ValueError(f"{manifest_path}: sources[{index}] must be an object")
            path_text = str(source.get("path") or "")
            if not path_text:
                raise ValueError(f"{manifest_path}: sources[{index}] missing path")
            _check_local_path(Path(path_text), label=f"candidate manifest sources[{index}]")
    else:
        source_path = summary.get("source_path")
        if not source_path:
            raise ValueError("request summary missing source_path")
        manifest = {
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
                    "source_id": "phase51z_unlinked_native_role_sources",
                    "venue_id": summary.get("venue_id", "all"),
                    "path": str(_check_local_path(Path(str(source_path)), label="source path")),
                }
            ],
            "source_links": [],
        }
    manifest = dict(manifest)
    manifest["baseline_commit"] = BASELINE_COMMIT
    manifest["no_live_flag"] = True
    manifest["approved_for_live"] = False
    manifest["approved_for_canary"] = False
    manifest["approved_for_model_training"] = False
    manifest["approved_for_capital_escalation"] = False
    manifest["admissible_for_financial_claim"] = False
    manifest["admissible_for_ev_admission"] = False
    manifest["live_orders_allowed"] = False
    manifest["capital_change_allowed"] = False
    manifest["risk_limit_relaxation_allowed"] = False
    manifest["source_links"] = [
        {
            "source_link_id": "phase51ad_materialized_source_links",
            "path": str(sidecar_path),
        }
    ]
    return manifest


def build_source_link_sidecar_materialization(
    *,
    request_pack: Path,
    mapping: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    request_pack = _check_local_path(request_pack, label="request-pack")
    mapping = _check_local_path(mapping, label="mapping")
    if not request_pack.is_dir():
        raise ValueError(f"request-pack must be a directory: {request_pack}")
    if not mapping.is_file():
        raise ValueError(f"mapping must be a file: {mapping}")
    run_id = _check_run_id(run_id or f"PHASE51AD-SOURCE-LINK-SIDECAR-MATERIALIZE-{_utc_stamp()}")
    output_root = _check_local_path(output_root or DEFAULT_OUTPUT_ROOT, label="output-root")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    summary_in = _load_summary(request_pack)
    request_sources = _load_request_sources(request_pack)
    by_group, by_order_key, request_targets = _load_request_targets(request_pack)
    sidecars, raw_labels = _load_mapping_rows(
        mapping_path=mapping,
        request_sources=request_sources,
        by_group=by_group,
        by_order_key=by_order_key,
    )

    sidecar_path = out_dir / "source_links.sanitized.jsonl"
    labels_path = out_dir / "phase51ad_source_link_sidecar_materialize_labels.jsonl"
    candidate_manifest_path = out_dir / "candidate_manifest_with_materialized_sidecar.json"
    summary_path = out_dir / "phase51ad_source_link_sidecar_materialize_summary.json"

    labels: list[dict[str, Any]] = []
    for seq, raw in enumerate(sorted(raw_labels, key=lambda item: item["source_record_sha256"])):
        out = _base_record(run_id, seq, timestamp_ns, "PHASE51AD_SOURCE_LINK_SIDECAR_MATERIALIZE_LABEL")
        out.update({key: value for key, value in raw.items() if key != "label_seq"})
        labels.append(out)

    candidate_manifest = _materialized_candidate_manifest(request_pack, summary_in, sidecar_path)
    _write_jsonl(sidecar_path, sidecars, sidecar_output=True)
    _write_jsonl(labels_path, labels)
    _write_json(candidate_manifest_path, candidate_manifest)

    materialized_source_hashes = {str(row["source_record_sha256"]) for row in sidecars}
    missing_source_count = max(len(request_sources) - len(materialized_source_hashes), 0)
    materialized_by_venue = _count_by_venue(labels)
    phase51v_validation_command = (
        "python3 tools/phase51v_forward_capture_bundle_readiness.py "
        f"--target-run {summary_in.get('target_run')} "
        f"--candidate-manifest {candidate_manifest_path} "
        "--output-root runs/phase51v_forward_capture_bundle_readiness "
        f"--run-id {run_id}-PHASE51V-MATERIALIZED-SIDECAR-HOLD "
        f"--timestamp-ns {timestamp_ns}"
    )
    output_summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51ad_source_link_sidecar_materialized_nonlive_hold",
        "no_live_flag": True,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_financial_claim": False,
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
        "clears_phase51_blockers": False,
        "request_pack": str(request_pack),
        "request_pack_sha256": _stable_hash(
            {
                "sources": _sha256_file(request_pack / "source_link_request_sources.jsonl"),
                "targets": _sha256_file(request_pack / "source_link_request_targets.jsonl"),
            }
        ),
        "mapping_path_hash": _stable_hash(str(mapping)),
        "mapping_sha256": _sha256_file(mapping),
        "source_link_request_source_count": len(request_sources),
        "source_link_request_target_count": len(request_targets),
        "materialized_source_link_count": len(sidecars),
        "materialized_source_link_counts_by_venue": materialized_by_venue,
        "missing_source_link_count": missing_source_count,
        "candidate_sidecar_complete": missing_source_count == 0,
        "source_links_sanitized_path": str(sidecar_path),
        "candidate_manifest_with_materialized_sidecar": str(candidate_manifest_path),
        "phase51v_validation_command": phase51v_validation_command,
        "next_required_action": "run_phase51v_against_materialized_candidate_manifest",
        "promotion_boundary": "Phase 5.1v target-ready counts only",
    }
    _write_json(summary_path, output_summary)

    artifacts = [sidecar_path, labels_path, candidate_manifest_path, summary_path]
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(
        artifact_index_path,
        {
            "schema_version": 1,
            "metadata": output_summary,
            "artifacts": _artifact_infos(out_dir, artifacts),
        },
    )
    manifest_path = out_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_utc": created_utc,
            "metadata": output_summary,
            "files": _artifact_infos(out_dir, [*artifacts, artifact_index_path]),
        },
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_source_link_sidecar_materialization(
            request_pack=args.request_pack,
            mapping=args.mapping,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51ad_source_link_sidecar_materialize: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ad_source_link_sidecar_materialize: wrote {out_dir}")
    print("phase51ad_source_link_sidecar_materialize: status HOLD (materialized sidecar only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
