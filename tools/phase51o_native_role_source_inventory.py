#!/usr/bin/env python3
"""Build Phase 5.1o venue-native maker/taker source inventory.

This HOLD-only tool inventories explicit venue-native role evidence that can be
joined to observed P_fill labels by canonical_group_id. It also records source
artifact availability where historical artifacts exist but cannot be joined
without raw identifier reprocessing. It never infers maker/taker from post-only
intent, strategy role, price position, or fee schedules.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51o_native_role_source_inventory"
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
ALLOWED_NATIVE_SOURCES = {
    "VENUE_NATIVE_FILL_FIELD",
    "VENUE_NATIVE_TRADE_JOIN",
    "VENUE_NATIVE_FEE_ROLE",
    "LIGHTER_TRADES_JSON",
    "HYPERLIQUID_CROSSED",
    "PARADEX_LIQUIDITY",
    "ASTER_ORDER_TRADE_UPDATE_M",
    "EXTENDED_ISTAKER",
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
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _role_counts(value: Any) -> dict[str, int]:
    counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
    if not isinstance(value, dict):
        return counts
    for key in counts:
        counts[key] = max(0, _safe_int(value.get(key)) or 0)
    return counts


def _known_count(counts: dict[str, int]) -> int:
    return int(counts.get("MAKER") or 0) + int(counts.get("TAKER") or 0)


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _venue_status_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        venue = str(record.get("venue_id") or "UNKNOWN")
        status = str(record.get("native_role_source_status") or "UNKNOWN")
        venue_counts = counts.setdefault(venue, {})
        venue_counts[status] = venue_counts.get(status, 0) + 1
    return {venue: dict(sorted(values.items())) for venue, values in sorted(counts.items())}


def _source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("maker_taker_attribution_source") or "UNKNOWN")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


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


def _load_observed_pfill(observed_pfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = observed_pfill_run / "pfill_outcome_summary.json"
    labels_path = observed_pfill_run / "pfill_order_labels.jsonl"
    summary = _load_json(summary_path)
    _check_unsafe(summary, summary_path, label="summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        labels.append(label)
    expected = _safe_int(summary.get("order_label_count"))
    if expected is not None and len(labels) != expected:
        raise ValueError(f"{labels_path} label count {len(labels)} != summary order_label_count {expected}")
    return summary, labels


def _native_source(row: dict[str, Any]) -> str:
    return str(row.get("maker_taker_attribution_source") or row.get("native_role_source") or "")


def _load_native_role_evidence(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for input_path in paths:
        path = _resolve_path(input_path)
        count = 0
        for line_no, row in _iter_jsonl(path):
            _check_unsafe(row, path, label="native role evidence")
            for raw_field in RAW_IDENTIFIER_FIELDS:
                if raw_field in row:
                    raise ValueError(f"{path}:{line_no} contains raw identifier field {raw_field}")
            canonical_group_id = str(row.get("canonical_group_id") or "")
            if not canonical_group_id:
                raise ValueError(f"{path}:{line_no} missing canonical_group_id")
            source = _native_source(row)
            if source not in ALLOWED_NATIVE_SOURCES:
                raise ValueError(f"{path}:{line_no} unsupported native role source {source!r}")
            if canonical_group_id in by_group:
                raise ValueError(f"{path}:{line_no} duplicate canonical_group_id={canonical_group_id}")
            by_group[canonical_group_id] = row
            count += 1
        inputs.append({"path": str(path), "sha256": _sha256_file(path), "native_role_record_count": count})
    return by_group, inputs


def _source_artifact_info(path: Path) -> tuple[str, str] | None:
    name = path.name.lower()
    if name == "trades_backfill.sanitized.json":
        return "LIGHTER_TRADES_JSON", "lighter"
    if name in {"lighter_raw_native_truth_labels.jsonl", "lighter_native_identity_gap_labels.jsonl"}:
        return "LIGHTER_NATIVE_TRUTH_LABELS", "lighter"
    return None


def _jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _safe_json_item_count(path: Path) -> int | None:
    if path.suffix.lower() == ".jsonl":
        return _jsonl_count(path)
    try:
        data = _load_json(path)
    except Exception:
        return None
    for key in ("trade_count", "label_count", "native_role_record_count"):
        value = _safe_int(data.get(key))
        if value is not None:
            return value
    if isinstance(data.get("trades"), list):
        return len(data["trades"])
    return None


def _scan_source_artifacts(source_roots: list[Path]) -> list[dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for source_root in source_roots:
        root = _resolve_path(source_root)
        if not root.exists():
            raise ValueError(f"source root does not exist: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            artifact_info = _source_artifact_info(path)
            if artifact_info is None:
                continue
            artifact_type, venue_id = artifact_info
            key = str(path)
            if key in artifacts:
                continue
            artifacts[key] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "artifact_type": artifact_type,
                "venue_id": venue_id,
                "item_count": _safe_json_item_count(path),
                "raw_identifier_redaction_status": "PASS_METADATA_ONLY",
                "canonical_join_status": "UNVERIFIED_OR_NOT_CANONICAL_JOINED",
            }
    return [artifacts[key] for key in sorted(artifacts)]


def build_native_role_source_inventory(
    *,
    observed_pfill_run: Path,
    source_roots: list[Path],
    native_role_jsonl: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51O-NATIVE-ROLE-SOURCE-INVENTORY-{_utc_stamp()}"
    output_root = _resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    observed_pfill_run = _resolve_path(observed_pfill_run)
    pfill_summary, pfill_labels = _load_observed_pfill(observed_pfill_run)
    native_by_group, native_inputs = _load_native_role_evidence(native_role_jsonl)
    source_artifacts = _scan_source_artifacts(source_roots)
    source_artifacts_by_venue: dict[str, list[dict[str, Any]]] = {}
    for artifact in source_artifacts:
        venue_id = str(artifact.get("venue_id") or "UNKNOWN")
        source_artifacts_by_venue.setdefault(venue_id, []).append(artifact)

    inventory_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    evidence_seq = 0
    for seq, label in enumerate(pfill_labels, start=1):
        canonical_group_id = str(label.get("canonical_group_id") or "")
        if not canonical_group_id:
            raise ValueError("observed P_fill label missing canonical_group_id")
        fill_count = _safe_int(label.get("fill_count")) or 0
        input_counts = _role_counts(label.get("maker_taker_role_counts"))
        native_record = native_by_group.get(canonical_group_id)
        status = "NO_FILL_NOT_APPLICABLE"
        hold_reason = "no_fill"
        source = "NO_FILL_NOT_APPLICABLE"
        effective_counts = {"MAKER": 0, "TAKER": 0, "UNKNOWN": 0}
        evidence_record_emitted = False
        if fill_count > 0:
            effective_counts = input_counts
            if _known_count(input_counts) >= fill_count and int(input_counts.get("UNKNOWN") or 0) == 0:
                status = "OBSERVED_PRESERVED"
                hold_reason = "input_already_has_complete_maker_taker_roles"
                source = "INPUT_PFILL_ROLE_COUNTS"
            elif native_record is not None:
                native_counts = _role_counts(native_record.get("maker_taker_role_counts"))
                source = _native_source(native_record)
                if _known_count(native_counts) >= fill_count and int(native_counts.get("UNKNOWN") or 0) == 0:
                    status = "EXACT_CANONICAL_NATIVE_ROLE_EVIDENCE"
                    hold_reason = "canonical_group_id_native_role_evidence_available"
                    effective_counts = native_counts
                    evidence_seq += 1
                    evidence = _base_record(
                        run_id,
                        evidence_seq,
                        timestamp_ns,
                        "PHASE51O_NATIVE_ROLE_EVIDENCE",
                    )
                    evidence.update({
                        "source": "phase51o_native_role_source_inventory",
                        "canonical_group_id": canonical_group_id,
                        "source_telemetry_sha256": label.get("source_telemetry_sha256"),
                        "venue_id": label.get("venue_id"),
                        "side": label.get("side"),
                        "fill_count": fill_count,
                        "maker_taker_role_counts": native_counts,
                        "maker_taker_attribution_source": source,
                        "native_role_source_status": status,
                        "native_role_source_hold_reason": hold_reason,
                    })
                    evidence_records.append(evidence)
                    evidence_record_emitted = True
                else:
                    status = "PARTIAL_CANONICAL_NATIVE_ROLE_EVIDENCE"
                    hold_reason = "canonical_group_id_native_role_evidence_incomplete"
                    effective_counts = native_counts
            elif source_artifacts_by_venue.get(str(label.get("venue_id") or "UNKNOWN")):
                status = "SOURCE_AVAILABLE_NO_CANONICAL_JOIN"
                hold_reason = "source_artifacts_exist_but_no_exact_canonical_group_native_role_evidence"
                source = "UNJOINED_SOURCE_ARTIFACTS"
            else:
                status = "MISSING_VENUE_NATIVE_ROLE_SOURCE"
                hold_reason = "requires_explicit_venue_native_fill_or_trade_role_evidence"
                source = "UNKNOWN_NO_NATIVE_ROLE_SOURCE"

        record = _base_record(run_id, seq, timestamp_ns, "PHASE51O_NATIVE_ROLE_SOURCE_INVENTORY_LABEL")
        record.update({
            "source": "phase51o_native_role_source_inventory",
            "observed_pfill_run": str(observed_pfill_run),
            "canonical_group_id": canonical_group_id,
            "canonical_order_key": label.get("order_key"),
            "source_telemetry_sha256": label.get("source_telemetry_sha256"),
            "venue_id": label.get("venue_id"),
            "side": label.get("side"),
            "fill_count": fill_count,
            "input_maker_taker_role_counts": input_counts,
            "effective_maker_taker_role_counts": effective_counts,
            "native_role_source_status": status,
            "native_role_source_hold_reason": hold_reason,
            "maker_taker_attribution_source": source,
            "exact_canonical_native_role_record_supplied": native_record is not None,
            "native_role_evidence_record_emitted": evidence_record_emitted,
            "source_artifacts_available_without_canonical_join": (
                bool(source_artifacts_by_venue.get(str(label.get("venue_id") or "UNKNOWN")))
                and status == "SOURCE_AVAILABLE_NO_CANONICAL_JOIN"
            ),
        })
        inventory_records.append(record)

    labels_path = out_dir / "native_role_source_inventory_labels.jsonl"
    evidence_path = out_dir / "native_role_evidence.jsonl"
    summary_path = out_dir / "native_role_source_inventory_summary.json"
    _write_jsonl(labels_path, inventory_records)
    _write_jsonl(evidence_path, evidence_records)

    status_counts = _status_counts(inventory_records, "native_role_source_status")
    missing = (
        status_counts.get("MISSING_VENUE_NATIVE_ROLE_SOURCE", 0)
        + status_counts.get("SOURCE_AVAILABLE_NO_CANONICAL_JOIN", 0)
        + status_counts.get("PARTIAL_CANONICAL_NATIVE_ROLE_EVIDENCE", 0)
    )
    gate_reason = (
        "phase51o_native_role_sources_complete"
        if missing == 0
        else "phase51o_native_role_sources_incomplete"
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "no_live_flag": True,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
        "observed_pfill_run": str(observed_pfill_run),
        "observed_pfill_summary_sha256": _sha256_file(observed_pfill_run / "pfill_outcome_summary.json"),
        "observed_pfill_labels_sha256": _sha256_file(observed_pfill_run / "pfill_order_labels.jsonl"),
        "source_roots": [str(_resolve_path(root)) for root in source_roots],
        "source_artifact_inventory": source_artifacts,
        "source_artifact_venues": sorted(source_artifacts_by_venue),
        "native_role_inputs": native_inputs,
        "label_count": len(inventory_records),
        "filled_count": sum(1 for record in inventory_records if int(record.get("fill_count") or 0) > 0),
        "input_observed_preserved_count": status_counts.get("OBSERVED_PRESERVED", 0),
        "recovered_native_role_count": status_counts.get("EXACT_CANONICAL_NATIVE_ROLE_EVIDENCE", 0),
        "native_role_evidence_record_count": len(evidence_records),
        "missing_native_role_source_count": status_counts.get("MISSING_VENUE_NATIVE_ROLE_SOURCE", 0),
        "source_available_no_canonical_join_count": status_counts.get("SOURCE_AVAILABLE_NO_CANONICAL_JOIN", 0),
        "partial_canonical_native_role_evidence_count": status_counts.get("PARTIAL_CANONICAL_NATIVE_ROLE_EVIDENCE", 0),
        "native_role_source_status_counts": status_counts,
        "native_role_source_status_counts_by_venue": _venue_status_counts(inventory_records),
        "maker_taker_attribution_source_counts": _source_counts(inventory_records),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, evidence_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, evidence_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", default=[])
    parser.add_argument("--native-role-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    source_roots = args.source_root or [ROOT / "runs"]
    try:
        out_dir = build_native_role_source_inventory(
            observed_pfill_run=args.observed_pfill_run,
            source_roots=source_roots,
            native_role_jsonl=args.native_role_jsonl,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51o_native_role_source_inventory: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51o_native_role_source_inventory: wrote {out_dir}")
    print("phase51o_native_role_source_inventory: status HOLD (native role inventory only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
