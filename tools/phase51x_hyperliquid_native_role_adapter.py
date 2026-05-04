#!/usr/bin/env python3
"""Build Phase 5.1x Hyperliquid native-role rows from local userFills snapshots.

This HOLD-only adapter consumes already-local Hyperliquid ``userFills`` style
JSON/JSONL snapshots and emits Phase 5.1v-ready source rows containing only the
native ``crossed`` maker/taker truth plus canonical join keys. It performs no
network access, reads no secrets, submits no orders, and never infers
maker/taker role from strategy intent.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51x_hyperliquid_native_role_adapter"
OFFICIAL_DOCS = [
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
]
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


def _path_text_is_unsafe(path_text: str) -> bool:
    lower = path_text.lower()
    first_part = lower.split("/", 1)[0]
    return lower.startswith(("http://", "https://", "http:/", "https:/")) or "://" in lower or ":" in first_part


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink path is prohibited: {candidate}")


def _check_local_source_path(path: Path) -> Path:
    if _path_text_is_unsafe(str(path)):
        raise ValueError(f"network source paths are prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as Hyperliquid native source input: {resolved}")
    _check_no_symlink(resolved)
    if not resolved.exists():
        raise ValueError(f"source path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"source path must be a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"source path must be .json or .jsonl: {resolved}")
    return resolved


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
    leaked = sorted(set(record) & ORDER_IDENTITY_FIELDS)
    if leaked:
        raise ValueError(f"{path} output leaked raw identifier fields: {leaked}")


def _payload_records(payload: Any, inherited: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    inherited = dict(inherited or {})
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_payload_records(item, inherited))
        return out
    if not isinstance(payload, dict):
        return []
    inherited_next = {
        **inherited,
        **{
            key: payload[key]
            for key in ("canonical_group_id", "order_key", "venue", "venue_id")
            if key in payload
        },
    }
    for key in SOURCE_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            out: list[dict[str, Any]] = []
            for item in value:
                out.extend(_payload_records(item, inherited_next))
            return out
    return [{**inherited, **payload}]


def _iter_source_records(paths: list[Path]):
    for path in paths:
        source_path = _check_local_source_path(path)
        if source_path.suffix == ".jsonl":
            for line_no, row in _iter_jsonl(source_path):
                _check_unsafe_flags(row, source_path, label="source row")
                _check_no_secret_fields(row, source_path, label="source row")
                for item in _payload_records(row):
                    yield source_path, line_no, item
            continue
        payload = _load_json(source_path)
        _check_unsafe_flags(payload, source_path, label="source payload")
        _check_no_secret_fields(payload, source_path, label="source payload")
        for line_no, row in enumerate(_payload_records(payload), start=1):
            if isinstance(row, dict):
                _check_unsafe_flags(row, source_path, label="source row")
                _check_no_secret_fields(row, source_path, label="source row")
                yield source_path, line_no, row


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


def _load_target_run(target_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_run = _resolve_path(target_run)
    summary_path = target_run / "phase51u_forward_capture_target_manifest_summary.json"
    targets_path = target_run / "native_role_capture_targets.jsonl"
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {summary_path}")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    _check_unsafe_flags(summary, summary_path, label="target summary")
    targets: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(targets_path):
        if str(row.get("venue_id") or "").lower() != "hyperliquid":
            continue
        _check_unsafe_flags(row, targets_path, label="target row")
        targets.append(row)
    return summary, targets


def _source_run_paths_from_targets(targets: list[dict[str, Any]], observed_by_group: dict[str, dict[str, Any]]) -> set[Path]:
    paths: set[Path] = set()
    for target in targets:
        observed = observed_by_group.get(str(target.get("canonical_group_id") or ""))
        if not observed:
            continue
        for value in observed.get("source_pfill_run_paths") or []:
            if value:
                paths.add(_resolve_path(Path(str(value))))
    return paths


def _load_source_pfill_by_order_key(paths: set[Path]) -> dict[str, dict[str, Any]]:
    by_order_key: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(paths):
        _, labels = _load_pfill_labels(run_dir)
        for label in labels:
            order_key = str(label.get("order_key") or "")
            if not order_key:
                raise ValueError(f"{run_dir} source P-fill row missing order_key")
            if order_key in by_order_key:
                raise ValueError(f"duplicate source order_key across source P-fill runs: {order_key}")
            by_order_key[order_key] = label
    return by_order_key


def _id_hashes(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    return {_stable_hash(value), _stable_hash(str(value))}


def _identity_hashes_from_label(label: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (label.get("order_id_hash"), label.get("client_order_id_hash"))
        if value
    }


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


def _target_identity_map(
    targets: list[dict[str, Any]],
    observed_labels: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], int]:
    observed_by_group = {str(label.get("canonical_group_id") or ""): label for label in observed_labels}
    source_paths = _source_run_paths_from_targets(targets, observed_by_group)
    source_by_order_key = _load_source_pfill_by_order_key(source_paths)
    targets_by_hash: dict[str, set[str]] = {}
    target_by_group: dict[str, dict[str, Any]] = {}
    for target in targets:
        group = str(target.get("canonical_group_id") or "")
        if not group:
            raise ValueError("Hyperliquid target row missing canonical_group_id")
        target_by_group[group] = target
        observed = observed_by_group.get(group)
        if observed:
            for identity_hash in _identity_hashes_from_label(observed):
                targets_by_hash.setdefault(identity_hash, set()).add(group)
            for source_order_key in observed.get("source_order_keys") or []:
                source_label = source_by_order_key.get(str(source_order_key))
                if not source_label:
                    continue
                for identity_hash in _identity_hashes_from_label(source_label):
                    targets_by_hash.setdefault(identity_hash, set()).add(group)
    return targets_by_hash, target_by_group, len(source_paths)


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


def build_hyperliquid_native_role_adapter(
    *,
    observed_pfill_run: Path,
    target_run: Path,
    source_json: list[Path],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    if not source_json:
        raise ValueError("at least one --source-json is required")
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_summary, observed_labels = _load_pfill_labels(observed_pfill_run)
    target_summary, targets = _load_target_run(target_run)
    targets_by_hash, target_by_group, source_pfill_run_count = _target_identity_map(targets, observed_labels)
    ambiguous_hashes = {identity_hash for identity_hash, groups in targets_by_hash.items() if len(groups) > 1}

    source_rows_seen = 0
    emitted_by_source_hash: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_artifacts: dict[str, dict[str, Any]] = {}
    for seq, (path, line_no, row) in enumerate(_iter_source_records(source_json), start=1):
        source_rows_seen += 1
        source_path_hash = _stable_hash(str(path))
        artifact = source_artifacts.setdefault(
            source_path_hash,
            {"path_hash": source_path_hash, "sha256": _sha256_file(path), "source_row_count": 0},
        )
        artifact["source_row_count"] += 1
        source_hash = _stable_hash(row)
        identity_hashes = _identity_hashes_from_source(row)
        matched_groups: set[str] = set()
        ambiguous_match_count = 0
        for identity_hash in identity_hashes:
            groups = targets_by_hash.get(identity_hash, set())
            if not groups:
                continue
            if identity_hash in ambiguous_hashes:
                ambiguous_match_count += 1
                continue
            matched_groups |= groups

        crossed = row.get("crossed")
        status = "NO_OBSERVED_IDENTITY_MATCH"
        group = ""
        if row.get("canonical_group_id") in target_by_group:
            group = str(row["canonical_group_id"])
            status = "SOURCE_ROW_CANONICAL_GROUP_MATCH"
        elif row.get("order_key"):
            for target_group, target in target_by_group.items():
                if row.get("order_key") == target.get("order_key"):
                    group = target_group
                    status = "SOURCE_ROW_ORDER_KEY_MATCH"
                    break
        elif not identity_hashes:
            status = "NO_ORDER_IDENTITY_HASH"
        elif ambiguous_match_count and not matched_groups:
            status = "AMBIGUOUS_OBSERVED_IDENTITY_MATCH"
        elif len(matched_groups) > 1:
            status = "AMBIGUOUS_OBSERVED_IDENTITY_MATCH"
        elif len(matched_groups) == 1:
            group = next(iter(matched_groups))
            status = "OBSERVED_IDENTITY_MATCH"
        if group and not isinstance(crossed, bool):
            status = "MATCHED_WITHOUT_CROSSED"
        if group and isinstance(crossed, bool) and source_hash not in emitted_by_source_hash:
            target = target_by_group[group]
            output = _base_record(run_id, len(output_rows), timestamp_ns, "PHASE51X_HYPERLIQUID_NATIVE_ROLE_SOURCE")
            output.update(
                {
                    "venue_id": "hyperliquid",
                    "canonical_group_id": group,
                    "order_key": target.get("order_key"),
                    "crossed": crossed,
                    "native_role_source": "HYPERLIQUID_CROSSED",
                    "source_record_sha256": source_hash,
                    "official_docs": OFFICIAL_DOCS,
                }
            )
            output_rows.append(output)
            emitted_by_source_hash.add(source_hash)
            status = "HYPERLIQUID_CROSSED_SOURCE_EMITTED"

        label = _base_record(run_id, seq, timestamp_ns, "PHASE51X_HYPERLIQUID_NATIVE_ROLE_ADAPTER_LABEL")
        label.update(
            {
                "source_path_hash": source_path_hash,
                "source_line": line_no,
                "source_record_sha256": source_hash,
                "source_identity_hash_count": len(identity_hashes),
                "matched_canonical_target_count": len(matched_groups) or int(bool(group)),
                "matched_ambiguous_identity_hash_count": ambiguous_match_count,
                "canonical_group_id": group or None,
                "crossed_present": isinstance(crossed, bool),
                "adapter_status": status,
            }
        )
        labels.append(label)

    output_path = out_dir / "hyperliquid_forward_native_role_snapshot.jsonl"
    labels_path = out_dir / "hyperliquid_native_role_adapter_labels.jsonl"
    summary_path = out_dir / "phase51x_hyperliquid_native_role_adapter_summary.json"
    manifest_path = out_dir / "phase51x_manifest.json"
    _write_jsonl(output_path, output_rows)
    _write_jsonl(labels_path, labels)

    recovered_groups = {str(row.get("canonical_group_id") or "") for row in output_rows}
    target_groups = set(target_by_group)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": (
            "phase51x_hyperliquid_native_role_adapter_complete_nonlive_hold"
            if target_groups and target_groups <= recovered_groups
            else "phase51x_hyperliquid_native_role_adapter_incomplete_nonlive_hold"
        ),
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
        "official_docs": OFFICIAL_DOCS,
        "observed_pfill_run": str(_resolve_path(observed_pfill_run)),
        "observed_pfill_gate_status": observed_summary.get("gate_status"),
        "target_run": str(_resolve_path(target_run)),
        "target_gate_status": target_summary.get("gate_status"),
        "hyperliquid_target_count": len(targets),
        "hyperliquid_target_recovered_count": len(target_groups & recovered_groups),
        "source_file_count": len(source_json),
        "source_pfill_run_count": source_pfill_run_count,
        "identity_hash_target_count": len(targets_by_hash),
        "ambiguous_identity_hash_target_count": len(ambiguous_hashes),
        "source_row_count": source_rows_seen,
        "source_row_emitted_count": len(output_rows),
        "adapter_status_counts": _status_counts(labels, "adapter_status"),
        "output_path": str(output_path),
        "clears_phase51_blockers": False,
        "source_artifacts": sorted(source_artifacts.values(), key=lambda item: item["path_hash"]),
    }
    _write_json(summary_path, summary)
    artifacts = [output_path, labels_path, summary_path]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": _artifact_infos(out_dir, artifacts),
        },
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51x_hyperliquid_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_hyperliquid_native_role_adapter(
            observed_pfill_run=args.observed_pfill_run,
            target_run=args.target_run,
            source_json=args.source_json,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51x_hyperliquid_native_role_adapter: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51x_hyperliquid_native_role_adapter: wrote {out_dir}")
    print("phase51x_hyperliquid_native_role_adapter: status HOLD (Hyperliquid native role source only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
