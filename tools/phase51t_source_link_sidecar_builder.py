#!/usr/bin/env python3
"""Build Phase 5.1s-compatible source-link sidecars from local snapshots.

This HOLD-only tool reads quarantined local venue-native snapshots and observed
P_fill labels, matches only by existing redacted order/client identifier hashes,
and emits a redacted source-link sidecar that can be staged through Phase 5.1s
and consumed by Phase 5.1r. It never infers maker/taker role or native-limit
pressure, never fetches network data, and never authorizes live execution.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51t_source_link_sidecar_builder"

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

ORDER_IDENTITY_FIELDS = {
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
    "venue_order_id",
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


def _is_uri_like(value: str) -> bool:
    return "://" in value or value.startswith(("http:", "https:", "s3:", "gs:"))


def _check_local_path(path: Path, *, label: str) -> Path:
    raw = str(path)
    if _is_uri_like(raw):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if resolved.suffix == ".env":
        raise ValueError(f"env files are prohibited as Phase 5.1t {label} inputs")
    if resolved.is_symlink():
        raise ValueError(f"symlink {label} path is prohibited: {resolved}")
    return resolved


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


def _check_unsafe_flags(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


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


def _iter_source_records(paths: list[Path]):
    for raw_path in paths:
        path = _check_local_path(raw_path, label="source")
        if path.is_dir():
            candidates = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}
            )
        else:
            candidates = [path]
        for source_path in candidates:
            source_path = _check_local_path(source_path, label="source")
            if source_path.suffix not in {".json", ".jsonl"}:
                continue
            if source_path.suffix == ".jsonl":
                for line_no, row in _iter_jsonl(source_path):
                    _check_unsafe_flags(row, source_path, label="source row")
                    _check_no_secret_fields(row, source_path, label="source row")
                    for item in _payload_records(row):
                        yield source_path, line_no, item
            else:
                payload = _load_json(source_path)
                if isinstance(payload, dict):
                    _check_unsafe_flags(payload, source_path, label="source payload")
                    _check_no_secret_fields(payload, source_path, label="source payload")
                for line_no, row in enumerate(_payload_records(payload), start=1):
                    _check_unsafe_flags(row, source_path, label="source row")
                    _check_no_secret_fields(row, source_path, label="source row")
                    yield source_path, line_no, row


def _load_hold_summary(run_dir: Path, filename: str) -> dict[str, Any]:
    path = _resolve_path(run_dir) / filename
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    if payload.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{path} baseline_commit mismatch")
    _check_unsafe_flags(payload, path, label="summary")
    return payload


def _load_pfill_labels(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = _resolve_path(run_dir)
    summary = _load_hold_summary(run_dir, "pfill_outcome_summary.json")
    labels_path = run_dir / "pfill_order_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe_flags(label, labels_path, label="pfill label")
        labels.append(label)
    expected = summary.get("order_label_count")
    if expected is not None and int(expected) != len(labels):
        raise ValueError(f"{labels_path} label count {len(labels)} != order_label_count {expected}")
    return summary, labels


def _source_run_paths_from_canonical(labels: list[dict[str, Any]]) -> set[Path]:
    paths: set[Path] = set()
    for label in labels:
        for value in label.get("source_pfill_run_paths") or []:
            if value:
                paths.add(_resolve_path(Path(str(value))))
    return paths


def _load_source_pfill_by_order_key(paths: set[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_order_key: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for run_dir in sorted(paths):
        summary, labels = _load_pfill_labels(run_dir)
        inputs.append({
            "run_path": str(run_dir),
            "run_id": summary.get("run_id"),
            "pfill_summary_sha256": _sha256_file(run_dir / "pfill_outcome_summary.json"),
            "pfill_labels_sha256": _sha256_file(run_dir / "pfill_order_labels.jsonl"),
        })
        for label in labels:
            order_key = str(label.get("order_key") or "")
            if not order_key:
                raise ValueError(f"{run_dir} source P_fill row missing order_key")
            if order_key in by_order_key:
                raise ValueError(f"duplicate source order_key across source P_fill runs: {order_key}")
            by_order_key[order_key] = label
    return by_order_key, inputs


def _identity_hashes_from_label(label: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (label.get("order_id_hash"), label.get("client_order_id_hash"))
        if value
    }


def _register_target(
    targets_by_hash: dict[str, set[tuple[str, str]]],
    identity_hash: str,
    canonical_group_id: str,
    order_key: str,
) -> None:
    if identity_hash:
        targets_by_hash.setdefault(identity_hash, set()).add((canonical_group_id, order_key))


def _build_identity_targets(
    canonical_labels: list[dict[str, Any]],
    source_by_order_key: dict[str, dict[str, Any]],
) -> dict[str, set[tuple[str, str]]]:
    targets_by_hash: dict[str, set[tuple[str, str]]] = {}
    for label in canonical_labels:
        canonical_group_id = str(label.get("canonical_group_id") or "")
        order_key = str(label.get("order_key") or "")
        if not canonical_group_id:
            raise ValueError("observed P_fill label missing canonical_group_id")
        for identity_hash in _identity_hashes_from_label(label):
            _register_target(targets_by_hash, identity_hash, canonical_group_id, order_key)
        for source_order_key in label.get("source_order_keys") or []:
            source_label = source_by_order_key.get(str(source_order_key))
            if not source_label:
                continue
            for identity_hash in _identity_hashes_from_label(source_label):
                _register_target(targets_by_hash, identity_hash, canonical_group_id, order_key)
    return targets_by_hash


def _id_hashes(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    return {_stable_hash(value), _stable_hash(str(value))}


def _identity_hashes_from_source(row: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(row, dict):
        for key, value in row.items():
            if str(key) in ORDER_IDENTITY_FIELDS:
                hashes.update(_id_hashes(value))
            hashes.update(_identity_hashes_from_source(value))
    elif isinstance(row, list):
        for value in row:
            hashes.update(_identity_hashes_from_source(value))
    return hashes


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


def _source_link_record(source_hash: str, canonical_group_id: str, order_key: str) -> dict[str, Any]:
    record = {
        "phase51s_source_record_sha256": source_hash,
        "canonical_group_id": canonical_group_id,
    }
    if order_key:
        record["order_key"] = order_key
    for flag in UNSAFE_TRUE_FLAGS:
        record.setdefault(flag, False)
    return record


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def build_source_link_sidecar(
    *,
    observed_pfill_run: Path,
    source_roots: list[Path],
    source_jsons: list[Path],
    source_pfill_runs: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    if not source_roots and not source_jsons:
        raise ValueError("at least one --source-root or --source-json is required")
    run_id = run_id or f"PHASE51T-SOURCE-LINK-SIDECAR-BUILDER-{_utc_stamp()}"
    output_root = _resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    observed_summary, canonical_labels = _load_pfill_labels(observed_pfill_run)
    source_paths = {_resolve_path(path) for path in source_pfill_runs}
    source_paths |= _source_run_paths_from_canonical(canonical_labels)
    source_by_order_key, source_pfill_inputs = _load_source_pfill_by_order_key(source_paths)
    targets_by_hash = _build_identity_targets(canonical_labels, source_by_order_key)

    labels: list[dict[str, Any]] = []
    source_links: list[dict[str, Any]] = []
    emitted_by_source_hash: dict[str, tuple[str, str]] = {}
    source_artifacts: dict[str, dict[str, Any]] = {}
    ambiguous_target_hashes = {
        identity_hash for identity_hash, targets in targets_by_hash.items() if len(targets) > 1
    }

    source_inputs = list(source_roots) + list(source_jsons)
    for seq, (source_path, source_line, row) in enumerate(_iter_source_records(source_inputs), start=1):
        source_path_hash = _stable_hash(str(source_path))
        artifact = source_artifacts.setdefault(
            source_path_hash,
            {
                "path_hash": source_path_hash,
                "sha256": _sha256_file(source_path),
                "source_row_count": 0,
            },
        )
        artifact["source_row_count"] += 1
        source_hash = _stable_hash(row)
        identity_hashes = _identity_hashes_from_source(row)
        matched_targets: set[tuple[str, str]] = set()
        matched_ambiguous_hash_count = 0
        for identity_hash in identity_hashes:
            targets = targets_by_hash.get(identity_hash, set())
            if not targets:
                continue
            if identity_hash in ambiguous_target_hashes:
                matched_ambiguous_hash_count += 1
                continue
            matched_targets |= targets

        status = "NO_OBSERVED_IDENTITY_MATCH"
        canonical_group_id = ""
        order_key = ""
        if row.get("canonical_group_id") or row.get("order_key"):
            status = "SOURCE_ROW_ALREADY_HAS_JOIN_KEY"
        elif not identity_hashes:
            status = "NO_ORDER_IDENTITY_HASH"
        elif matched_ambiguous_hash_count and not matched_targets:
            status = "AMBIGUOUS_OBSERVED_IDENTITY_MATCH"
        elif len(matched_targets) > 1:
            status = "AMBIGUOUS_OBSERVED_IDENTITY_MATCH"
        elif len(matched_targets) == 1:
            canonical_group_id, order_key = next(iter(matched_targets))
            existing = emitted_by_source_hash.get(source_hash)
            if existing is None:
                emitted_by_source_hash[source_hash] = (canonical_group_id, order_key)
                source_links.append(_source_link_record(source_hash, canonical_group_id, order_key))
                status = "SOURCE_LINK_EMITTED"
            elif existing == (canonical_group_id, order_key):
                status = "DUPLICATE_SOURCE_HASH_ALREADY_EMITTED"
            else:
                raise ValueError("same source-record hash mapped to conflicting canonical targets")

        label = _base_record(run_id, seq, timestamp_ns, "PHASE51T_SOURCE_LINK_SIDECAR_BUILDER_LABEL")
        label.update({
            "source": "phase51t_source_link_sidecar_builder",
            "source_record_sha256": source_hash,
            "source_path_hash": source_path_hash,
            "source_line": source_line,
            "source_identity_hash_count": len(identity_hashes),
            "matched_canonical_target_count": len(matched_targets),
            "matched_ambiguous_identity_hash_count": matched_ambiguous_hash_count,
            "canonical_group_id": canonical_group_id or None,
            "order_key": order_key or None,
            "source_link_status": status,
        })
        labels.append(label)

    source_links_path = out_dir / "source_links.sanitized.jsonl"
    labels_path = out_dir / "source_link_builder_labels.jsonl"
    summary_path = out_dir / "phase51t_source_link_sidecar_builder_summary.json"
    _write_jsonl(source_links_path, source_links)
    _write_jsonl(labels_path, labels)

    source_row_count = len(labels)
    emitted_count = len(source_links)
    gate_reason = (
        "phase51t_source_link_sidecar_builder_complete_nonlive_hold"
        if emitted_count > 0
        else "phase51t_source_link_sidecar_builder_no_links_emitted"
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
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
        "clears_phase51_blockers": False,
        "downstream_tool": "tools/phase51s_local_native_source_acquisition.py",
        "downstream_manifest_source_links_path": "source_links.sanitized.jsonl",
        "observed_pfill_run": str(_resolve_path(observed_pfill_run)),
        "observed_pfill_gate_status": observed_summary.get("gate_status"),
        "observed_pfill_gate_reason": observed_summary.get("gate_reason"),
        "observed_pfill_label_count": len(canonical_labels),
        "source_pfill_inputs": source_pfill_inputs,
        "identity_hash_target_count": len(targets_by_hash),
        "ambiguous_identity_hash_target_count": len(ambiguous_target_hashes),
        "source_file_count": len(source_artifacts),
        "source_row_count": source_row_count,
        "source_link_record_count": emitted_count,
        "source_link_status_counts": _status_counts(labels, "source_link_status"),
        "source_artifacts": sorted(source_artifacts.values(), key=lambda item: item["path_hash"]),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [source_links_path, labels_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [source_links_path, labels_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", default=[])
    parser.add_argument("--source-json", type=Path, action="append", default=[])
    parser.add_argument("--source-pfill-run", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_source_link_sidecar(
            observed_pfill_run=args.observed_pfill_run,
            source_roots=args.source_root,
            source_jsons=args.source_json,
            source_pfill_runs=args.source_pfill_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51t_source_link_sidecar_builder: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51t_source_link_sidecar_builder: wrote {out_dir}")
    print("phase51t_source_link_sidecar_builder: status HOLD (source-link sidecars only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
