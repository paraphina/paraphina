#!/usr/bin/env python3
"""Audit whether existing sanitized source-link sidecars satisfy a request pack.

This HOLD-only tool performs no network access, reads no secrets, and does not
infer source links. It compares Phase 5.1z request-pack source hashes against
existing local `source_links.sanitized.jsonl` files, then emits a deterministic
reuse/missing-hash report for the next capture/review step.
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
DEFAULT_REQUEST_PACK = (
    ROOT
    / "runs/phase51z_source_link_request_pack"
    / "PHASE51Z-ALLVENUE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ac_source_link_reuse_audit"

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

SOURCE_HASH_FIELDS = {
    "phase51s_source_record_sha256",
    "source_record_sha256",
    "redacted_source_record_sha256",
}

SOURCE_LINK_ALLOWED_FIELDS = (
    SOURCE_HASH_FIELDS
    | {"canonical_group_id", "order_key", "source_link_id", "source_link_sha256"}
    | UNSAFE_TRUE_FLAGS
)


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
        raise ValueError(f"env files are prohibited as Phase 5.1ac {label} inputs")
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
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


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


def _source_hashes(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for field in sorted(SOURCE_HASH_FIELDS):
        value = row.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"source hash field {field} must be a string")
        if value:
            out.append(value)
    return sorted(set(out))


def _count_by_venue(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        venue = str(row.get("venue_id") or "unknown").lower()
        counts[venue] = counts.get(venue, 0) + 1
    return dict(sorted(counts.items()))


def _request_pack_path(request_pack: Path, file_name: str) -> Path:
    path = request_pack / file_name
    if not path.exists():
        raise ValueError(f"missing request-pack file: {path}")
    return path


def _load_request_sources(request_pack: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    request_sources_path = _request_pack_path(request_pack, "source_link_request_sources.jsonl")
    request_sources: list[dict[str, Any]] = []
    request_by_hash: dict[str, dict[str, Any]] = {}
    for line_no, row in _iter_jsonl(request_sources_path):
        _check_unsafe_flags(row, request_sources_path, label="request source")
        _check_no_secret_fields(row, request_sources_path, label="request source")
        _check_no_raw_identifier_fields(row, request_sources_path, label="request source")
        source_hash = row.get("source_record_sha256")
        if not isinstance(source_hash, str) or not source_hash:
            raise ValueError(f"{request_sources_path}:{line_no} request source missing source_record_sha256")
        if source_hash in request_by_hash:
            raise ValueError(f"{request_sources_path}:{line_no} duplicate request source hash {source_hash}")
        request_sources.append(dict(row))
        request_by_hash[source_hash] = dict(row)
    if not request_sources:
        raise ValueError("request pack has no source rows")
    return request_sources, request_by_hash


def _load_request_targets(request_pack: Path) -> list[dict[str, Any]]:
    target_path = _request_pack_path(request_pack, "source_link_request_targets.jsonl")
    targets: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(target_path):
        _check_unsafe_flags(row, target_path, label="request target")
        _check_no_secret_fields(row, target_path, label="request target")
        _check_no_raw_identifier_fields(row, target_path, label="request target")
        targets.append(dict(row))
    return targets


def _discover_sidecars(sidecar_root: Path, request_pack: Path) -> list[Path]:
    sidecar_root = _check_local_path(sidecar_root, label="sidecar-root")
    request_pack = request_pack.resolve()
    if sidecar_root.is_file():
        return [sidecar_root]
    if not sidecar_root.exists():
        raise ValueError(f"sidecar-root does not exist: {sidecar_root}")
    candidates = sorted(sidecar_root.rglob("source_links.sanitized.jsonl"))
    return [
        path
        for path in candidates
        if path.resolve() != request_pack / "source_links.proposed.empty.jsonl"
    ]


def _load_sidecar_rows(sidecar_paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for sidecar_path in sidecar_paths:
        sidecar_path = _check_local_path(sidecar_path, label="sidecar")
        if sidecar_path.suffix != ".jsonl":
            raise ValueError(f"sidecar path must be a .jsonl file: {sidecar_path}")
        row_count = 0
        for line_no, row in _iter_jsonl(sidecar_path):
            _check_unsafe_flags(row, sidecar_path, label="sidecar row")
            _check_no_secret_fields(row, sidecar_path, label="sidecar row")
            _check_no_raw_identifier_fields(row, sidecar_path, label="sidecar row")
            unexpected = sorted(set(row) - SOURCE_LINK_ALLOWED_FIELDS)
            if unexpected:
                raise ValueError(f"{sidecar_path}:{line_no} unsupported sidecar fields: {unexpected}")
            source_hashes = _source_hashes(row)
            if not source_hashes:
                raise ValueError(f"{sidecar_path}:{line_no} sidecar row missing source hash")
            if not row.get("canonical_group_id") and not row.get("order_key"):
                raise ValueError(f"{sidecar_path}:{line_no} sidecar row missing canonical_group_id or order_key")
            for source_hash in source_hashes:
                rows.append(
                    {
                        "source_hash": source_hash,
                        "source_link_row_sha256": _stable_hash(row),
                        "source_link_file": str(sidecar_path),
                        "source_link_file_sha256": _sha256_file(sidecar_path),
                        "canonical_group_id": row.get("canonical_group_id"),
                        "order_key": row.get("order_key"),
                    }
                )
            row_count += 1
        inventory.append(
            {
                "source_link_file": str(sidecar_path),
                "source_link_file_sha256": _sha256_file(sidecar_path),
                "source_link_row_count": row_count,
            }
        )
    return rows, inventory


def build_source_link_reuse_audit(
    *,
    request_pack: Path,
    sidecar_root: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    request_pack = _check_local_path(request_pack, label="request-pack")
    if not request_pack.is_dir():
        raise ValueError(f"request-pack must be a directory: {request_pack}")
    run_id = _check_run_id(run_id or f"PHASE51AC-SOURCE-LINK-REUSE-AUDIT-{_utc_stamp()}")
    output_root = _check_local_path(output_root or DEFAULT_OUTPUT_ROOT, label="output-root")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    request_sources, request_by_hash = _load_request_sources(request_pack)
    request_targets = _load_request_targets(request_pack)
    sidecar_paths = _discover_sidecars(sidecar_root, request_pack)
    sidecar_rows, sidecar_inventory = _load_sidecar_rows(sidecar_paths)

    matched_hashes: set[str] = set()
    reusable_rows: list[dict[str, Any]] = []
    seq = 0
    for row in sidecar_rows:
        source_hash = row["source_hash"]
        if source_hash not in request_by_hash:
            continue
        request_source = request_by_hash[source_hash]
        out = _base_record(run_id, seq, timestamp_ns, "PHASE51AC_REUSABLE_SOURCE_LINK")
        seq += 1
        out.update(
            {
                "source_record_sha256": source_hash,
                "venue_id": str(request_source.get("venue_id") or "unknown").lower(),
                "source_link_row_sha256": row["source_link_row_sha256"],
                "source_link_file_sha256": row["source_link_file_sha256"],
                "canonical_group_id": row.get("canonical_group_id"),
                "order_key": row.get("order_key"),
                "reuse_status": "REUSABLE_EXISTING_SOURCE_LINK",
            }
        )
        matched_hashes.add(source_hash)
        reusable_rows.append(out)

    missing_rows: list[dict[str, Any]] = []
    for request_source in request_sources:
        source_hash = str(request_source.get("source_record_sha256") or "")
        if source_hash in matched_hashes:
            continue
        out = _base_record(run_id, seq, timestamp_ns, "PHASE51AC_MISSING_SOURCE_LINK")
        seq += 1
        out.update(
            {
                "source_record_sha256": source_hash,
                "venue_id": str(request_source.get("venue_id") or "unknown").lower(),
                "reuse_status": "NO_REUSABLE_EXISTING_SOURCE_LINK",
            }
        )
        missing_rows.append(out)

    inventory_rows: list[dict[str, Any]] = []
    for item in sidecar_inventory:
        out = _base_record(run_id, seq, timestamp_ns, "PHASE51AC_SOURCE_LINK_INVENTORY")
        seq += 1
        out.update(
            {
                "source_link_file_path_hash": _stable_hash(item["source_link_file"]),
                "source_link_file_sha256": item["source_link_file_sha256"],
                "source_link_row_count": item["source_link_row_count"],
            }
        )
        inventory_rows.append(out)

    reusable_path = out_dir / "reusable_source_links.jsonl"
    missing_path = out_dir / "missing_source_link_request_sources.jsonl"
    inventory_path = out_dir / "source_link_sidecar_inventory.jsonl"
    summary_path = out_dir / "phase51ac_source_link_reuse_audit_summary.json"

    _write_jsonl(reusable_path, reusable_rows)
    _write_jsonl(missing_path, missing_rows)
    _write_jsonl(inventory_path, inventory_rows)

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51ac_source_link_reuse_audit_nonlive_hold",
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
        "source_link_request_source_count": len(request_sources),
        "source_link_request_source_counts_by_venue": _count_by_venue(request_sources),
        "source_link_request_target_count": len(request_targets),
        "source_link_request_target_counts_by_venue": _count_by_venue(request_targets),
        "existing_sidecar_file_count": len(sidecar_inventory),
        "existing_sidecar_row_count": sum(int(row["source_link_row_count"]) for row in sidecar_inventory),
        "reusable_source_link_count": len(reusable_rows),
        "reusable_source_link_counts_by_venue": _count_by_venue(reusable_rows),
        "missing_source_link_count": len(missing_rows),
        "missing_source_link_counts_by_venue": _count_by_venue(missing_rows),
        "candidate_sidecar_complete": len(missing_rows) == 0 and len(reusable_rows) > 0,
        "next_required_artifact": "validated_redacted_source_link_sidecar_or_direct_target_linkable_source",
        "promotion_boundary": "Phase 5.1v target-ready counts only",
    }
    _write_json(summary_path, summary)

    artifacts = [reusable_path, missing_path, inventory_path, summary_path]
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(
        artifact_index_path,
        {
            "schema_version": 1,
            "metadata": summary,
            "artifacts": _artifact_infos(out_dir, artifacts),
        },
    )
    manifest_path = out_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "created_utc": created_utc,
            "metadata": summary,
            "files": _artifact_infos(out_dir, [*artifacts, artifact_index_path]),
        },
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, default=DEFAULT_REQUEST_PACK)
    parser.add_argument("--sidecar-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_source_link_reuse_audit(
            request_pack=args.request_pack,
            sidecar_root=args.sidecar_root,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51ac_source_link_reuse_audit: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ac_source_link_reuse_audit: wrote {out_dir}")
    print("phase51ac_source_link_reuse_audit: status HOLD (reuse audit only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
