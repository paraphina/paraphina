#!/usr/bin/env python3
"""Build a HOLD-only source-link request pack for Phase 5.1z unlinked rows.

This tool does not infer source links. It packages sanitized unlinked
source-record hashes and current Phase 5.1u targets so a separate reviewer or
collector can produce a redacted source-link sidecar. The output remains
non-live and cannot clear Phase 5.1 readiness by itself.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51z_source_link_request_pack"

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
    "oid",
    "order_id",
    "order_id_str",
    "orderId",
    "raw_client_order_id",
    "raw_order_id",
    "trade_id",
    "tradeId",
    "venue_order_id",
}

TARGET_FIELDS = (
    "canonical_group_id",
    "order_key",
    "venue_id",
    "required_native_role_source",
    "required_native_role_fields",
    "first_fill_time_ms",
    "last_fill_time_ms",
    "side",
    "price",
    "size",
    "order_id_hash",
    "client_order_id_hash",
    "decision_id_hash",
    "source_telemetry_sha256",
    "source_order_key_count",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _is_uri_like(value: str) -> bool:
    return "://" in value or value.startswith(("http:", "https:", "s3:", "gs:"))


def _check_local_path(path: Path, *, label: str) -> Path:
    raw = str(path)
    if _is_uri_like(raw):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if resolved.suffix == ".env":
        raise ValueError(f"env files are prohibited as Phase 5.1z request-pack {label} inputs")
    if resolved.is_symlink():
        raise ValueError(f"symlink {label} path is prohibited: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                raise ValueError(f"{path} contains secret-shaped {label} field {key!r}")


def _check_no_raw_identifier_fields(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        unexpected = sorted(str(key) for key in obj if str(key) in RAW_IDENTIFIER_FIELDS)
        if unexpected:
            raise ValueError(f"{path} contains raw identifier {label} fields: {unexpected}")


def _check_unsafe_flags(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


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


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _load_target_run(target_run: Path, venue_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_run = _check_local_path(target_run, label="target-run")
    summary_path = target_run / "phase51u_forward_capture_target_manifest_summary.json"
    target_path = target_run / "native_role_capture_targets.jsonl"
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {summary_path}")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    _check_unsafe_flags(summary, summary_path, label="target summary")
    targets: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(target_path):
        _check_unsafe_flags(row, target_path, label="target row")
        _check_no_secret_fields(row, target_path, label="target row")
        if str(row.get("venue_id") or "").lower() != venue_id:
            continue
        out = _base_record("target_placeholder", len(targets), 0, "PHASE51Z_SOURCE_LINK_REQUEST_TARGET")
        out.update({key: row.get(key) for key in TARGET_FIELDS if key in row})
        out["accepted_source_hash_field"] = "source_record_sha256"
        out["sidecar_required"] = True
        out["source_link_status"] = "REQUESTED"
        targets.append(out)
    return summary, targets


def _source_path_from_run(source_run: Path) -> Path:
    return source_run / "source_snapshots" / "phase51z_forward_native_role_rows.jsonl"


def _load_source_rows(source_path: Path, venue_id: str) -> list[dict[str, Any]]:
    source_path = _check_local_path(source_path, label="source")
    sources: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for _, row in _iter_jsonl(source_path):
        _check_unsafe_flags(row, source_path, label="source row")
        _check_no_secret_fields(row, source_path, label="source row")
        _check_no_raw_identifier_fields(row, source_path, label="source row")
        if str(row.get("venue_id") or "").lower() != venue_id:
            continue
        if row.get("label_type") != "PHASE51Z_UNLINKED_NATIVE_ROLE_SOURCE":
            continue
        if row.get("canonical_group_id") or row.get("order_key"):
            continue
        source_hash = str(row.get("source_record_sha256") or "")
        if not source_hash:
            raise ValueError(f"{source_path} unlinked row missing source_record_sha256")
        if source_hash in seen_hashes:
            raise ValueError(f"{source_path} duplicate unlinked source_record_sha256 {source_hash}")
        seen_hashes.add(source_hash)
        sources.append(dict(row))
    return sources


def build_source_link_request_pack(
    *,
    target_run: Path,
    source_run: Path | None,
    source_json: Path | None,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    venue_id: str,
) -> Path:
    venue_id = venue_id.lower()
    if venue_id != "lighter":
        raise ValueError("Phase 5.1z source-link request pack currently supports venue_id=lighter only")
    if bool(source_run) == bool(source_json):
        raise ValueError("provide exactly one of --source-run or --source-json")
    run_id = run_id or f"PHASE51Z-SOURCE-LINK-REQUEST-PACK-{_utc_stamp()}"
    output_root = _resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    target_summary, target_rows = _load_target_run(_resolve_path(target_run), venue_id)
    source_path = _source_path_from_run(_resolve_path(source_run)) if source_run else _resolve_path(source_json)  # type: ignore[arg-type]
    source_rows = _load_source_rows(source_path, venue_id)
    if not source_rows:
        raise ValueError("no sanitized unlinked source rows found")
    if not target_rows:
        raise ValueError("no matching Phase 5.1u target rows found")

    request_sources: list[dict[str, Any]] = []
    for seq, row in enumerate(source_rows):
        out = _base_record(run_id, seq, timestamp_ns, "PHASE51Z_SOURCE_LINK_REQUEST_SOURCE")
        out.update(row)
        out["label_type"] = "PHASE51Z_SOURCE_LINK_REQUEST_SOURCE"
        out["source_link_status"] = "UNLINKED_SOURCE_REQUIRES_VALIDATED_SIDECAR"
        out["sidecar_required"] = True
        request_sources.append(out)

    request_targets: list[dict[str, Any]] = []
    for seq, row in enumerate(target_rows):
        out = dict(row)
        out.update(_base_record(run_id, seq, timestamp_ns + len(request_sources), "PHASE51Z_SOURCE_LINK_REQUEST_TARGET"))
        out.update({key: row.get(key) for key in TARGET_FIELDS if key in row})
        out["accepted_source_hash_field"] = "source_record_sha256"
        out["source_link_status"] = "TARGET_REQUIRES_VALIDATED_SOURCE_HASH"
        out["sidecar_required"] = True
        request_targets.append(out)

    source_output_path = out_dir / "source_link_request_sources.jsonl"
    target_output_path = out_dir / "source_link_request_targets.jsonl"
    empty_sidecar_path = out_dir / "source_links.proposed.empty.jsonl"
    candidate_manifest_path = out_dir / "candidate_manifest_with_empty_sidecar.json"
    sidecar_schema_path = out_dir / "source_link_sidecar_schema.json"
    request_md_path = out_dir / "REQUEST.md"
    summary_path = out_dir / "phase51z_source_link_request_pack_summary.json"

    _write_jsonl(source_output_path, request_sources)
    _write_jsonl(target_output_path, request_targets)
    _write_jsonl(empty_sidecar_path, [])
    _write_json(sidecar_schema_path, {
        "schema_version": 1,
        "required_fields": ["source_record_sha256"],
        "required_one_of": ["canonical_group_id", "order_key"],
        "allowed_fields": [
            "source_record_sha256",
            "phase51s_source_record_sha256",
            "redacted_source_record_sha256",
            "canonical_group_id",
            "order_key",
            *sorted(UNSAFE_TRUE_FLAGS),
        ],
        "unsafe_true_flags_must_be_false": sorted(UNSAFE_TRUE_FLAGS),
        "raw_identifier_fields_prohibited": sorted(RAW_IDENTIFIER_FIELDS),
        "promotion_boundary": "Phase 5.1v target-ready counts only",
    })
    _write_json(candidate_manifest_path, {
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
                "source_id": "phase51z_unlinked_lighter_native_role_sources",
                "venue_id": venue_id,
                "path": str(source_path),
            }
        ],
        "source_links": [
            {
                "source_link_id": "empty_placeholder_replace_with_validated_redacted_sidecar",
                "path": str(empty_sidecar_path),
            }
        ],
    })
    validation_command = (
        "python3 tools/phase51v_forward_capture_bundle_readiness.py "
        f"--target-run {target_run} "
        f"--candidate-manifest {candidate_manifest_path} "
        "--output-root runs/phase51v_forward_capture_bundle_readiness "
        f"--run-id {run_id}-PHASE51V-EMPTY-SIDECAR-HOLD "
        f"--timestamp-ns {timestamp_ns}"
    )
    request_md_path.write_text(
        "\n".join(
            [
                "# Phase 5.1z Lighter Source-Link Request Pack",
                "",
                "Status: HOLD. This pack requests a validated redacted source-link sidecar.",
                "",
                "Required sidecar fields: `source_record_sha256` plus `canonical_group_id` or `order_key`.",
                "Do not include raw order IDs, client IDs, trade IDs, secrets, or unsafe true flags.",
                "The empty sidecar included here is a placeholder and must not be treated as readiness.",
                "",
                "Validation command after replacing the empty sidecar path in the candidate manifest:",
                "",
                "```bash",
                validation_command,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51z_source_link_request_pack_nonlive_hold",
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
        "clears_phase51_blockers": False,
        "venue_id": venue_id,
        "target_run": str(_resolve_path(target_run)),
        "target_run_gate_status": target_summary.get("gate_status"),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "source_link_request_source_count": len(request_sources),
        "source_link_request_target_count": len(request_targets),
        "source_link_sidecar_template_row_count": 0,
        "candidate_manifest_with_empty_sidecar": str(candidate_manifest_path),
        "phase51v_validation_command": validation_command,
        "next_required_artifact": "validated_redacted_source_link_sidecar",
        "promotion_boundary": "Phase 5.1v target-ready counts only",
    }
    _write_json(summary_path, summary)

    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    artifacts = [
        source_output_path,
        target_output_path,
        empty_sidecar_path,
        candidate_manifest_path,
        sidecar_schema_path,
        request_md_path,
        summary_path,
    ]
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, artifacts),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [*artifacts, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--source-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--venue-id", default="lighter")
    args = parser.parse_args()
    try:
        out_dir = build_source_link_request_pack(
            target_run=args.target_run,
            source_run=args.source_run,
            source_json=args.source_json,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            venue_id=args.venue_id,
        )
    except Exception as exc:
        print(f"phase51z_source_link_request_pack: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51z_source_link_request_pack: wrote {out_dir}")
    print("phase51z_source_link_request_pack: status HOLD (source-link request only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
