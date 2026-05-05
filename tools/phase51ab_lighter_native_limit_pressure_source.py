#!/usr/bin/env python3
"""Stage sanitized Lighter native-limit pressure rows for Phase 5.1v.

This HOLD-only tool performs no network access and does not call sendTx,
sendTxBatch, nextNonce, or any venue write path. It accepts only local
sanitized JSONL rows that already contain observed Lighter native-limit
pressure fields, rejects raw identifiers/secrets/unsafe flags, and emits a
Phase 5.1v-ready candidate manifest.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ab_lighter_native_limit_pressure_source"

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

BOOL_FIELDS = (
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
)

STRING_FIELDS = {
    "schema_version",
    "label_type",
    "label_seq",
    "timestamp_local_ns",
    "run_id",
    "baseline_commit",
    "gate_status",
    "raw_identifier_redaction_status",
    "venue_id",
    "canonical_group_id",
    "order_key",
    "source_record_sha256",
    "redacted_source_record_sha256",
    "phase51s_source_record_sha256",
    "native_limit_event_time_status",
    "native_limit_source_sha256",
    "native_limit_pressure_source",
    "pressure_capture_mode",
    "pressure_source_type",
    "source_event_time_ms",
    "snapshot_ts_ms",
}

INT_FIELDS = {
    "active_order_headroom_account",
    "active_order_headroom_market",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "rest_requests_per_minute_limit",
    "rest_requests_per_minute_remaining",
    "weighted_requests_per_minute_limit",
    "weighted_requests_per_minute_remaining",
}

FLOAT_FIELDS = {"native_limit_staleness_ms"}

ALLOWED_SOURCE_FIELDS = {
    "schema_version",
    "label_type",
    "label_seq",
    "timestamp_local_ns",
    "run_id",
    "baseline_commit",
    "gate_status",
    "no_live_flag",
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
    "raw_identifier_redaction_status",
    "venue_id",
    "canonical_group_id",
    "order_key",
    "source_record_sha256",
    "redacted_source_record_sha256",
    "phase51s_source_record_sha256",
    "active_order_headroom_account",
    "active_order_headroom_market",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "rest_requests_per_minute_limit",
    "rest_requests_per_minute_remaining",
    "weighted_requests_per_minute_limit",
    "weighted_requests_per_minute_remaining",
    "native_limit_event_time_status",
    "native_limit_staleness_ms",
    "native_limit_source_sha256",
    "native_limit_pressure_source",
    "pressure_capture_mode",
    "pressure_source_type",
    "source_event_time_ms",
    "snapshot_ts_ms",
}

EVENT_TIME_OK = {"EVENT_TIME_ALIGNED"}


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
            raise ValueError(f"symlink source path is prohibited: {candidate}")


def _check_local_path(path: Path, *, label: str) -> Path:
    raw = str(path)
    if _is_uri_like(raw):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as Phase 5.1ab {label} inputs")
    _check_no_symlink(resolved)
    if label == "input-jsonl" and resolved.suffix != ".jsonl":
        raise ValueError(f"Phase 5.1ab {label} input must be a .jsonl file")
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
                raise ValueError(f"{path} contains secret-shaped {label} field {key!r}")


def _check_no_raw_identifier_fields(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = sorted(str(key) for key in obj if str(key) in RAW_IDENTIFIER_FIELDS)
        if raw_fields:
            raise ValueError(f"{path} contains raw identifier {label} fields: {raw_fields}")


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


def _validate_source_field_types(row: dict[str, Any], path: Path) -> None:
    for key in sorted(set(row) & set(BOOL_FIELDS)):
        if not isinstance(row[key], bool):
            raise ValueError(f"{path} source row field {key} must be boolean")
    for key in sorted(set(row) & STRING_FIELDS):
        if row[key] is not None and not isinstance(row[key], (str, int)):
            raise ValueError(f"{path} source row field {key} must be scalar")
    for key in sorted(set(row) & INT_FIELDS):
        if row[key] is not None and _safe_int(row[key]) is None:
            raise ValueError(f"{path} source row field {key} must be int-compatible")
    for key in sorted(set(row) & FLOAT_FIELDS):
        if row[key] is not None and _safe_float(row[key]) is None:
            raise ValueError(f"{path} source row field {key} must be float-compatible")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
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


def _row_complete(row: dict[str, Any]) -> bool:
    has_active = (
        _safe_int(row.get("active_order_headroom_account")) is not None
        and _safe_int(row.get("active_order_headroom_market")) is not None
    )
    has_sendtx = (
        _safe_int(row.get("sendtx_per_minute_limit")) is not None
        and _safe_int(row.get("sendtx_per_minute_remaining")) is not None
    )
    has_rest = (
        _safe_int(row.get("rest_requests_per_minute_limit")) is not None
        and _safe_int(row.get("rest_requests_per_minute_remaining")) is not None
    )
    has_weighted = (
        _safe_int(row.get("weighted_requests_per_minute_limit")) is not None
        and _safe_int(row.get("weighted_requests_per_minute_remaining")) is not None
    )
    return has_active and has_sendtx and (has_rest or has_weighted) and row.get("native_limit_event_time_status") in EVENT_TIME_OK


def _normalize_row(row: dict[str, Any], *, run_id: str, seq: int, timestamp_ns: int, source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _check_unsafe_flags(row, source_path, label="source row")
    _check_no_secret_fields(row, source_path, label="source row")
    _check_no_raw_identifier_fields(row, source_path, label="source row")
    unexpected = sorted(set(row) - ALLOWED_SOURCE_FIELDS)
    if unexpected:
        raise ValueError(f"{source_path} source row has unsupported fields: {unexpected}")
    _validate_source_field_types(row, source_path)
    if str(row.get("venue_id") or "").lower() != "lighter":
        raise ValueError(f"{source_path} source row venue_id must be lighter")
    if not row.get("canonical_group_id") and not row.get("order_key") and not (
        row.get("source_record_sha256")
        or row.get("redacted_source_record_sha256")
        or row.get("phase51s_source_record_sha256")
    ):
        raise ValueError(f"{source_path} source row needs a direct join key or redacted source hash")

    out = _base_record(run_id, seq, timestamp_ns, "PHASE51AB_LIGHTER_NATIVE_LIMIT_PRESSURE_SOURCE")
    for key in sorted(ALLOWED_SOURCE_FIELDS):
        if key in row and key not in out:
            out[key] = row[key]
    out["venue_id"] = "lighter"
    out.setdefault("native_limit_pressure_source", "SANITIZED_EXTERNAL_PRESSURE_SOURCE")
    out["native_limit_source_sha256"] = str(row.get("native_limit_source_sha256") or _stable_hash(row))
    out["native_limit_staleness_ms"] = _safe_float(row.get("native_limit_staleness_ms"))
    for key in (
        "active_order_headroom_account",
        "active_order_headroom_market",
        "sendtx_per_minute_limit",
        "sendtx_per_minute_remaining",
        "rest_requests_per_minute_limit",
        "rest_requests_per_minute_remaining",
        "weighted_requests_per_minute_limit",
        "weighted_requests_per_minute_remaining",
    ):
        out[key] = _safe_int(row.get(key))
    complete = _row_complete(out)
    label = _base_record(run_id, seq, timestamp_ns, "PHASE51AB_LIGHTER_NATIVE_LIMIT_PRESSURE_LABEL")
    label.update(
        {
            "venue_id": "lighter",
            "source_record_sha256": out.get("source_record_sha256") or out.get("redacted_source_record_sha256") or out.get("phase51s_source_record_sha256"),
            "canonical_group_id": out.get("canonical_group_id"),
            "order_key": out.get("order_key"),
            "native_limit_complete_source_row": complete,
            "native_limit_event_time_status": out.get("native_limit_event_time_status"),
            "has_active_order_headroom": out.get("active_order_headroom_account") is not None and out.get("active_order_headroom_market") is not None,
            "has_sendtx_pressure": out.get("sendtx_per_minute_limit") is not None and out.get("sendtx_per_minute_remaining") is not None,
            "has_rest_or_weighted_pressure": (
                out.get("rest_requests_per_minute_limit") is not None
                and out.get("rest_requests_per_minute_remaining") is not None
            )
            or (
                out.get("weighted_requests_per_minute_limit") is not None
                and out.get("weighted_requests_per_minute_remaining") is not None
            ),
        }
    )
    return out, label


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def build_lighter_native_limit_pressure_source(
    *,
    input_jsonl: Path,
    target_run: Path | None,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = _check_run_id(run_id or f"PHASE51AB-LIGHTER-NATIVE-LIMIT-PRESSURE-SOURCE-{_utc_stamp()}")
    output_root = _check_local_path(output_root or DEFAULT_OUTPUT_ROOT, label="output-root")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)
    input_path = _check_local_path(input_jsonl, label="input-jsonl")

    source_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for seq, (_, row) in enumerate(_iter_jsonl(input_path), start=1):
        source_row, label = _normalize_row(row, run_id=run_id, seq=seq, timestamp_ns=timestamp_ns, source_path=input_path)
        source_rows.append(source_row)
        labels.append(label)
    if not source_rows:
        raise ValueError("no Lighter native-limit pressure rows supplied")

    source_path = out_dir / "lighter_forward_native_limit_pressure_snapshot.jsonl"
    labels_path = out_dir / "lighter_native_limit_pressure_labels.jsonl"
    candidate_manifest_path = out_dir / "phase51v_candidate_manifest.json"
    summary_path = out_dir / "phase51ab_lighter_native_limit_pressure_summary.json"

    _write_jsonl(source_path, source_rows)
    _write_jsonl(labels_path, labels)
    _write_json(
        candidate_manifest_path,
        {
            "manifest_version": 1,
            "baseline_commit": BASELINE_COMMIT,
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
            "sources": [
                {
                    "source_id": "phase51ab_lighter_native_limit_pressure",
                    "venue_id": "lighter",
                    "path": str(source_path),
                }
            ],
            "source_links": [],
        },
    )

    complete_count = sum(1 for row in source_rows if _row_complete(row))
    target_run_path = _check_local_path(target_run, label="target-run") if target_run else None
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51ab_lighter_native_limit_pressure_source_nonlive_hold",
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
        "input_jsonl": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "source_row_count": len(source_rows),
        "complete_lighter_native_limit_source_row_count": complete_count,
        "candidate_manifest_path": str(candidate_manifest_path),
        "source_path": str(source_path),
        "labels_path": str(labels_path),
        "target_run": str(target_run_path) if target_run_path else None,
        "phase51v_validation_command": (
            "python3 tools/phase51v_forward_capture_bundle_readiness.py "
            f"--target-run {target_run_path} "
            f"--candidate-manifest {candidate_manifest_path} "
            "--output-root runs/phase51v_forward_capture_bundle_readiness "
            f"--run-id {run_id}-PHASE51V-HOLD "
            f"--timestamp-ns {timestamp_ns}"
            if target_run_path
            else None
        ),
        "promotion_boundary": "Phase 5.1v target-ready counts only",
    }
    _write_json(summary_path, summary)

    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    artifacts = [source_path, labels_path, candidate_manifest_path, summary_path]
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
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_lighter_native_limit_pressure_source(
            input_jsonl=args.input_jsonl,
            target_run=args.target_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51ab_lighter_native_limit_pressure_source: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ab_lighter_native_limit_pressure_source: wrote {out_dir}")
    print("phase51ab_lighter_native_limit_pressure_source: status HOLD (non-live pressure source only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
