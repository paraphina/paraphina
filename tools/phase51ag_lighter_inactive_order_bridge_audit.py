#!/usr/bin/env python3
"""Audit whether Lighter inactive orders can bridge native trade rows to targets.

This HOLD-only utility consumes only already-local sanitized artifacts. It never
calls network endpoints, never emits raw identifiers, and never infers links
from price/time/size. A bridge is accepted only when a sanitized inactive-order
hash uniquely maps to a current Phase 5.1 target and the same hash appears on a
native Lighter trade source row whose source_record_sha256 is present in the
current source-link request pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase51z_readonly_native_role_capture as phase51z


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ag_lighter_inactive_order_bridge_audit"

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
    "authorization",
    "auth_token",
    "bearer",
    "jwt",
    "password",
    "private_key",
    "secret",
    "signature",
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
    "client_order_id",
    "cloid",
    "cursor",
    "id",
    "next_cursor",
    "oid",
    "order_id",
    "raw_client_order_id",
    "raw_order_id",
    "trade_id",
    "tx_hash",
    "venue_order_id",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _field_looks_secret(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if "nonsecret" in normalized:
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


def _check_no_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            normalized = str(key).replace("-", "_").lower()
            if normalized in RAW_IDENTIFIER_FIELDS and not normalized.endswith("_sha256"):
                raise ValueError(f"{path} has raw identifier-shaped {label} field {key!r}")


def _check_output_safe(record: Any, path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


def _check_local_trade_input_safe(record: Any, path: Path) -> None:
    _check_unsafe_flags(record, path, label="trade input")
    _check_no_secret_fields(record, path, label="trade input")


def _check_local_file(path: Path, *, label: str) -> Path:
    if "://" in str(path):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if any(part == ".env" or part.endswith(".env") for part in resolved.parts):
        raise ValueError(f"env files are prohibited as {label} inputs")
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} path does not exist or is not a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"{label} path must be .json or .jsonl: {resolved}")
    return resolved


def _artifact_infos(root_dir: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _load_request_pack(request_pack: Path) -> tuple[dict[str, set[str]], set[str]]:
    targets_by_hash: dict[str, set[str]] = {}
    request_source_hashes: set[str] = set()
    target_path = request_pack / "source_link_request_targets.jsonl"
    source_path = request_pack / "source_link_request_sources.jsonl"
    for _, target in _iter_jsonl(target_path):
        _check_output_safe(target, target_path)
        if target.get("baseline_commit") != BASELINE_COMMIT:
            raise ValueError("request target baseline_commit mismatch")
        if target.get("venue_id") != "lighter":
            continue
        target_id = str(target.get("canonical_group_id") or target.get("order_key") or "")
        if not target_id:
            continue
        for hashed in (target.get("order_id_hash"), target.get("client_order_id_hash")):
            if isinstance(hashed, str) and len(hashed) == 64:
                targets_by_hash.setdefault(hashed.lower(), set()).add(target_id)
    for _, source in _iter_jsonl(source_path):
        _check_output_safe(source, source_path)
        if source.get("baseline_commit") != BASELINE_COMMIT:
            raise ValueError("request source baseline_commit mismatch")
        if source.get("venue_id") == "lighter" and isinstance(source.get("source_record_sha256"), str):
            request_source_hashes.add(str(source["source_record_sha256"]).lower())
    return targets_by_hash, request_source_hashes


def _inactive_order_target_hashes(inactive_orders_json: Path, targets_by_hash: dict[str, set[str]]) -> dict[str, str]:
    payload = _load_json(inactive_orders_json)
    _check_output_safe(payload, inactive_orders_json)
    matched: dict[str, str] = {}
    if isinstance(payload, dict) and isinstance(payload.get("orders"), list):
        rows = [row for row in payload["orders"] if isinstance(row, dict)]
    else:
        rows = phase51z._payload_records(payload)
    for row in rows:
        if not isinstance(row, dict):
            continue
        for hashed in (row.get("order_id_sha256"), row.get("client_order_id_sha256")):
            if not isinstance(hashed, str):
                continue
            target_ids = targets_by_hash.get(hashed.lower())
            if target_ids and len(target_ids) == 1:
                matched[hashed.lower()] = next(iter(target_ids))
    return matched


def _trade_bridge_rows(
    trade_source_json: Path,
    *,
    request_source_hashes: set[str],
    inactive_hash_to_target: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_json(trade_source_json)
    _check_local_trade_input_safe(payload, trade_source_json)
    records = phase51z._payload_records(payload)
    bridge_rows: list[dict[str, Any]] = []
    direct_candidate_hit_count = 0
    source_in_request_count = 0
    native_ready_count = 0
    for row in records:
        hashes, native_payload, _field_status = phase51z._extract_lighter_source(row)
        source_hash = phase51z._stable_hash(row).lower()
        if source_hash in request_source_hashes:
            source_in_request_count += 1
        if native_payload is not None:
            native_ready_count += 1
        for hashed in hashes:
            if hashed in inactive_hash_to_target:
                direct_candidate_hit_count += 1
                if native_payload is not None and source_hash in request_source_hashes:
                    bridge_rows.append({
                        "source_record_sha256": source_hash,
                        "canonical_group_id": inactive_hash_to_target[hashed],
                    })
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bridge_rows:
        unique[(row["source_record_sha256"], row["canonical_group_id"])] = row
    summary = {
        "path": str(trade_source_json),
        "trade_source_row_count": len(records),
        "native_ready_count": native_ready_count,
        "request_source_hash_overlap_count": source_in_request_count,
        "inactive_hash_candidate_hit_count": direct_candidate_hit_count,
        "bridge_row_count": len(unique),
        "bridge_target_count": len({row["canonical_group_id"] for row in unique.values()}),
        "bridge_source_count": len({row["source_record_sha256"] for row in unique.values()}),
    }
    return list(unique.values()), summary


def run(
    *,
    request_pack: Path,
    inactive_orders_json: Path,
    trade_source_json: list[Path],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    request_pack = _check_local_file(request_pack / "manifest.json", label="request pack manifest").parent
    inactive_orders_json = _check_local_file(inactive_orders_json, label="inactive orders")
    trade_sources = [_check_local_file(path, label="trade source") for path in trade_source_json]
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_by_hash, request_source_hashes = _load_request_pack(request_pack)
    inactive_hash_to_target = _inactive_order_target_hashes(inactive_orders_json, targets_by_hash)
    all_bridge_rows: list[dict[str, Any]] = []
    trade_source_summaries: list[dict[str, Any]] = []
    for path in trade_sources:
        rows, summary = _trade_bridge_rows(
            path,
            request_source_hashes=request_source_hashes,
            inactive_hash_to_target=inactive_hash_to_target,
        )
        all_bridge_rows.extend(rows)
        trade_source_summaries.append(summary)

    unique_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in all_bridge_rows:
        unique_rows[(row["source_record_sha256"], row["canonical_group_id"])] = row
    source_links = list(unique_rows.values())
    sidecar_path = out_dir / "source_links.proposed.sanitized.jsonl"
    summary_path = out_dir / "phase51ag_lighter_inactive_order_bridge_audit_summary.json"
    manifest_path = out_dir / "manifest.json"
    _write_jsonl(sidecar_path, source_links)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "lighter_inactive_order_bridge_audit_nonlive_hold",
        "request_pack": str(request_pack),
        "inactive_orders_json": str(inactive_orders_json),
        "trade_source_json": [str(path) for path in trade_sources],
        "lighter_request_source_hash_count": len(request_source_hashes),
        "lighter_target_hash_count": len(targets_by_hash),
        "inactive_order_target_hash_match_count": len(inactive_hash_to_target),
        "inactive_order_target_match_count": len(set(inactive_hash_to_target.values())),
        "trade_source_summaries": trade_source_summaries,
        "bridge_source_link_count": len(source_links),
        "bridge_target_count": len({row["canonical_group_id"] for row in source_links}),
        "bridge_source_count": len({row["source_record_sha256"] for row in source_links}),
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
    }
    _write_json(summary_path, summary)
    artifact_paths = [sidecar_path, summary_path]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_utc": summary["created_utc"],
            "metadata": summary,
            "files": _artifact_infos(out_dir, artifact_paths),
        },
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, required=True)
    parser.add_argument("--inactive-orders-json", type=Path, required=True)
    parser.add_argument("--trade-source-json", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AG-LIGHTER-INACTIVE-ORDER-BRIDGE-AUDIT-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = run(
            request_pack=args.request_pack,
            inactive_orders_json=args.inactive_orders_json,
            trade_source_json=args.trade_source_json,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns or int(datetime.now(timezone.utc).timestamp() * 1_000_000_000),
        )
    except Exception as exc:
        print(f"phase51ag_lighter_inactive_order_bridge_audit: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ag_lighter_inactive_order_bridge_audit: wrote {out_dir}")
    print("phase51ag_lighter_inactive_order_bridge_audit: status HOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
