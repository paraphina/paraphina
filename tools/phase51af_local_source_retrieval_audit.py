#!/usr/bin/env python3
"""Phase 5.1af local source retrieval audit.

This HOLD-only utility answers one narrow question: can existing local files
produce the missing Phase 5.1 source-link mapping or Lighter event-time native
limit pressure without inference?

It does not read env files, does not use network paths, does not emit raw venue
identifiers, and does not infer source links from time/price/size proximity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_REQUEST_PACK = (
    ROOT
    / "runs/phase51z_source_link_request_pack"
    / "PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51af_local_source_retrieval_audit"

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

JOIN_FIELDS = {"canonical_group_id", "order_key"}
SOURCE_HASH_FIELDS = {
    "phase51s_source_record_sha256",
    "redacted_source_record_sha256",
    "source_record_sha256",
}
NATIVE_ROLE_FIELD_NAMES = {
    "isTaker",
    "is_taker",
    "isMakerAsk",
    "is_maker_ask",
    "liquidity",
    "maker",
    "m",
}
RAW_ID_FIELD_NAMES = {
    "askClientId",
    "ask_client_id",
    "ask_id",
    "askId",
    "bidClientId",
    "bid_client_id",
    "bid_id",
    "bidId",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "current_client_order_id",
    "current_order_id",
    "decision_id",
    "externalId",
    "externalOrderId",
    "order_id",
    "orderId",
    "source_decision_id",
}
LIGHTER_PRESSURE_FIELD_NAMES = {
    "active_order_headroom_account",
    "active_order_headroom_market",
    "native_active_order_headroom_account",
    "native_active_order_headroom_market",
    "native_sendtx_per_minute_remaining",
    "rest_requests_per_minute_limit",
    "rest_requests_per_minute_remaining",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "weighted_requests_per_minute_limit",
    "weighted_requests_per_minute_remaining",
}
LOG_PATTERNS = (
    "sendtxbatch",
    "sendtx",
    "x-ratelimit",
    "rate_limit",
    "ratelimit",
    "remaining",
    "istaker",
    "liquidity",
    "clientorderid",
    "externalorderid",
    "order_trade_update",
    "ask_id",
    "bid_id",
    "is_maker_ask",
    "ismakerask",
    "weighted",
    "active_orders",
    "account_limits",
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
        raise ValueError(f"env files are prohibited as Phase 5.1af {label} inputs")
    _check_no_symlink(resolved)
    return resolved


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


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


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")


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


def _iter_json_objects_from_line(path: Path, line_no: int, line: str):
    text = line.strip()
    if not text:
        return
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            record, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}:{exc.pos + 1}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        yield record
        idx = end


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            for record in _iter_json_objects_from_line(path, line_no, line):
                yield line_no, record


def _base_record(run_id: str, label_seq: int, timestamp_ns: int, label_type: str) -> dict[str, Any]:
    return {
        "admissible_for_ev_admission": False,
        "admissible_for_financial_claim": False,
        "admissible_for_model_training": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_live": False,
        "approved_for_model_training": False,
        "baseline_commit": BASELINE_COMMIT,
        "capital_change_allowed": False,
        "gate_status": "HOLD",
        "label_seq": label_seq,
        "label_type": label_type,
        "live_orders_allowed": False,
        "no_live_flag": True,
        "raw_identifier_redaction_status": "PASS",
        "risk_limit_relaxation_allowed": False,
        "run_id": run_id,
        "schema_version": 1,
        "timestamp_local_ns": timestamp_ns,
    }


def _count_key_presence(value: Any, field_names: set[str], *, prefix: str = "") -> Counter[str]:
    counts: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in field_names:
                counts[path] += 1
            counts.update(_count_key_presence(child, field_names, prefix=path))
    elif isinstance(value, list):
        for child in value:
            counts.update(_count_key_presence(child, field_names, prefix=f"{prefix}[]"))
    return counts


def _scan_request_pack(request_pack: Path) -> dict[str, Any]:
    request_pack = _check_local_path(request_pack, label="request-pack")
    summary_path = request_pack / "phase51z_source_link_request_pack_summary.json"
    sources_path = request_pack / "source_link_request_sources.jsonl"
    targets_path = request_pack / "source_link_request_targets.jsonl"
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path} must contain a JSON object")
    _check_unsafe_flags(summary, summary_path, label="request-pack summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("request-pack baseline_commit mismatch")

    source_rows = 0
    target_rows = 0
    source_counts_by_venue: Counter[str] = Counter()
    target_counts_by_venue: Counter[str] = Counter()
    source_join_field_rows = 0
    source_hash_rows = 0
    native_role_field_counts: Counter[str] = Counter()

    for _, row in _iter_jsonl(sources_path):
        _check_unsafe_flags(row, sources_path, label="request source")
        _check_no_secret_fields(row, sources_path, label="request source")
        source_rows += 1
        source_counts_by_venue[str(row.get("venue_id") or "UNKNOWN")] += 1
        if any(row.get(field) for field in JOIN_FIELDS):
            source_join_field_rows += 1
        if any(row.get(field) for field in SOURCE_HASH_FIELDS):
            source_hash_rows += 1
        native_role_field_counts.update(_count_key_presence(row, NATIVE_ROLE_FIELD_NAMES))

    for _, row in _iter_jsonl(targets_path):
        _check_unsafe_flags(row, targets_path, label="request target")
        _check_no_secret_fields(row, targets_path, label="request target")
        target_rows += 1
        target_counts_by_venue[str(row.get("venue_id") or "UNKNOWN")] += 1

    return {
        "request_pack": str(request_pack),
        "request_summary_sha256": _sha256_file(summary_path),
        "source_rows": source_rows,
        "source_counts_by_venue": dict(sorted(source_counts_by_venue.items())),
        "source_hash_rows": source_hash_rows,
        "source_join_field_rows": source_join_field_rows,
        "source_join_fields_present": source_join_field_rows > 0,
        "target_rows": target_rows,
        "target_counts_by_venue": dict(sorted(target_counts_by_venue.items())),
        "native_role_field_presence_counts": dict(sorted(native_role_field_counts.items())),
        "next_required_artifact": summary.get("next_required_artifact"),
    }


def _parse_bounded_telemetry_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--bounded-telemetry must be EXPECTED_SHA256=PATH")
    expected, raw_path = spec.split("=", 1)
    expected = expected.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("--bounded-telemetry expected hash must be lowercase SHA256 hex")
    return expected, _check_local_path(Path(raw_path), label="bounded-telemetry")


def _scan_bounded_telemetry(specs: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in specs:
        expected_hash, path = _parse_bounded_telemetry_spec(spec)
        actual_hash = _sha256_file(path)
        record_count = 0
        raw_id_counts: Counter[str] = Counter()
        native_role_counts: Counter[str] = Counter()
        pressure_counts: Counter[str] = Counter()
        list_counts: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                for row in _iter_json_objects_from_line(path, line_no, line):
                    _check_unsafe_flags(row, path, label="bounded telemetry")
                    record_count += 1
                    for key, value in row.items():
                        if isinstance(value, list) and value:
                            list_counts[str(key)] += 1
                    raw_id_counts.update(_count_key_presence(row, RAW_ID_FIELD_NAMES))
                    native_role_counts.update(_count_key_presence(row, NATIVE_ROLE_FIELD_NAMES))
                    pressure_counts.update(_count_key_presence(row, LIGHTER_PRESSURE_FIELD_NAMES))
        results.append(
            {
                "path": str(path),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "sha256_matches_expected": actual_hash == expected_hash,
                "record_count": record_count,
                "list_presence_counts": dict(sorted(list_counts.items())),
                "raw_identifier_field_presence_count": sum(raw_id_counts.values()),
                "raw_identifier_field_presence_paths": sorted(raw_id_counts),
                "native_role_field_presence_count": sum(native_role_counts.values()),
                "native_role_field_presence_paths": sorted(native_role_counts),
                "lighter_pressure_field_presence_count": sum(pressure_counts.values()),
                "lighter_pressure_field_presence_paths": sorted(pressure_counts),
            }
        )
    return results


def _scan_logs(paths: list[Path], max_log_bytes: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pattern_bytes = [(pattern, pattern.lower().encode("utf-8")) for pattern in LOG_PATTERNS]
    for raw_path in paths:
        path = _check_local_path(raw_path, label="log")
        size = path.stat().st_size
        if size > max_log_bytes:
            results.append(
                {
                    "path": str(path),
                    "bytes": size,
                    "scan_status": "SKIPPED_MAX_BYTES",
                    "max_log_bytes": max_log_bytes,
                    "pattern_counts": {},
                }
            )
            continue
        counts: Counter[str] = Counter()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                lower = chunk.lower()
                for pattern, encoded in pattern_bytes:
                    count = lower.count(encoded)
                    if count:
                        counts[pattern] += count
        results.append(
            {
                "path": str(path),
                "bytes": size,
                "scan_status": "SCANNED",
                "max_log_bytes": max_log_bytes,
                "pattern_counts": dict(sorted(counts.items())),
            }
        )
    return results


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(root_dir).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, default=DEFAULT_REQUEST_PACK)
    parser.add_argument("--bounded-telemetry", action="append", default=[], help="EXPECTED_SHA256=PATH")
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--max-log-bytes", type=int, default=100_000_000)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AF-LOCAL-SOURCE-RETRIEVAL-AUDIT-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=time.time_ns())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_id = _check_run_id(args.run_id)
        if args.max_log_bytes < 0:
            raise ValueError("--max-log-bytes must be non-negative")
        output_root = _check_local_path(args.output_root, label="output-root")
        run_dir = output_root / run_id
        request_audit = _scan_request_pack(args.request_pack)
        bounded_audits = _scan_bounded_telemetry(args.bounded_telemetry)
        log_audits = _scan_logs(args.log, args.max_log_bytes)

        bounded_hashes_match = all(item["sha256_matches_expected"] for item in bounded_audits)
        source_rows_have_join_fields = bool(request_audit["source_join_fields_present"])
        source_link_complete_candidate = (
            source_rows_have_join_fields
            and request_audit["source_join_field_rows"] >= request_audit["target_rows"]
            and request_audit["target_rows"] > 0
        )
        bounded_has_lighter_pressure = any(item["lighter_pressure_field_presence_count"] > 0 for item in bounded_audits)
        logs_have_lighter_pressure = any(
            any(
                key in item["pattern_counts"]
                for key in ("sendtx", "sendtxbatch", "remaining", "x-ratelimit", "weighted")
            )
            for item in log_audits
            if item["scan_status"] == "SCANNED"
        )
        log_scans_complete = all(item["scan_status"] == "SCANNED" for item in log_audits)
        source_link_retrieval_status = (
            "COMPLETE_PHASE51AD_MAPPING_CANDIDATE"
            if source_link_complete_candidate
            else "DISCOVERY_HINT_ONLY"
            if source_rows_have_join_fields
            else "MISSING_REQUIRED_LINKAGE"
        )
        lighter_pressure_retrieval_status = (
            "DISCOVERY_HINT_ONLY"
            if bounded_has_lighter_pressure or logs_have_lighter_pressure
            else "MISSING_REQUIRED_PRESSURE_FIELDS"
        )
        runtime_log_pattern_status = (
            "DISCOVERY_HINT_ONLY" if logs_have_lighter_pressure else "NO_USABLE_PRESSURE_PATTERN"
        )
        local_retrieval_possible = source_link_complete_candidate

        labels: list[dict[str, Any]] = []
        label = _base_record(run_id, len(labels), args.timestamp_ns, "PHASE51AF_REQUEST_PACK_AUDIT")
        label.update(request_audit)
        labels.append(label)
        for item in bounded_audits:
            label = _base_record(run_id, len(labels), args.timestamp_ns, "PHASE51AF_BOUNDED_TELEMETRY_AUDIT")
            label.update(item)
            labels.append(label)
        for item in log_audits:
            label = _base_record(run_id, len(labels), args.timestamp_ns, "PHASE51AF_RUNTIME_LOG_AUDIT")
            label.update(item)
            labels.append(label)
        verdict = _base_record(run_id, len(labels), args.timestamp_ns, "PHASE51AF_LOCAL_SOURCE_RETRIEVAL_VERDICT")
        verdict.update(
            {
                "local_retrieval_possible_without_inference": local_retrieval_possible,
                "bounded_telemetry_hashes_match": bounded_hashes_match,
                "source_rows_have_join_fields": source_rows_have_join_fields,
                "source_link_retrieval_status": source_link_retrieval_status,
                "bounded_telemetry_has_lighter_pressure_fields": bounded_has_lighter_pressure,
                "lighter_pressure_retrieval_status": lighter_pressure_retrieval_status,
                "runtime_logs_have_lighter_pressure_patterns": logs_have_lighter_pressure,
                "runtime_log_pattern_status": runtime_log_pattern_status,
                "runtime_log_scans_complete": log_scans_complete,
                "clears_phase51_blockers": False,
                "verdict": "HOLD",
                "next_required_artifacts": [
                    "validated_redacted_source_link_mapping",
                    "sanitized_lighter_event_time_native_limit_pressure_rows",
                ],
            }
        )
        labels.append(verdict)

        summary = {
            "admissible_for_ev_admission": False,
            "admissible_for_financial_claim": False,
            "approved_for_canary": False,
            "approved_for_capital_escalation": False,
            "approved_for_live": False,
            "approved_for_model_training": False,
            "baseline_commit": BASELINE_COMMIT,
            "bounded_telemetry_audits": bounded_audits,
            "bounded_telemetry_hashes_match": bounded_hashes_match,
            "capital_change_allowed": False,
            "clears_phase51_blockers": False,
            "created_utc": _timestamp_ns_to_utc(args.timestamp_ns),
            "gate_reason": "phase51af_local_sources_do_not_supply_missing_mapping_or_pressure",
            "gate_status": "HOLD",
            "live_orders_allowed": False,
            "local_retrieval_possible_without_inference": local_retrieval_possible,
            "log_audits": log_audits,
            "no_live_flag": True,
            "raw_identifier_redaction_status": "PASS",
            "request_pack_audit": request_audit,
            "risk_limit_relaxation_allowed": False,
            "run_id": run_id,
            "runtime_log_scans_complete": log_scans_complete,
            "schema_version": 1,
            "source_rows_have_join_fields": source_rows_have_join_fields,
            "source_link_retrieval_status": source_link_retrieval_status,
            "lighter_pressure_retrieval_status": lighter_pressure_retrieval_status,
            "runtime_log_pattern_status": runtime_log_pattern_status,
            "next_single_move": (
                "obtain validated redacted source-link mapping and sanitized "
                "Lighter event-time pressure rows; then run Phase 5.1ad/ab -> 5.1ae -> 5.1v"
            ),
        }

        summary_path = run_dir / "phase51af_local_source_retrieval_audit_summary.json"
        labels_path = run_dir / "phase51af_local_source_retrieval_audit_labels.jsonl"
        manifest_path = run_dir / "manifest.json"
        _write_json(summary_path, summary)
        _write_jsonl(labels_path, labels)
        manifest = {
            "created_utc": _timestamp_ns_to_utc(args.timestamp_ns),
            "files": _artifact_infos(run_dir, [summary_path, labels_path]),
            "metadata": summary,
            "schema_version": 1,
        }
        _write_json(manifest_path, manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should fail closed with clear stderr.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
