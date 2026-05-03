#!/usr/bin/env python3
"""Build Phase 5.1n venue-native maker/taker attribution recovery labels.

This HOLD-only tool reconciles observed P_fill labels against optional
venue-native role evidence. It never infers maker/taker from post-only flags,
order purpose, expected strategy behavior, or fee schedules.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51n_maker_taker_attribution_recovery"
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


def _load_native_role_evidence(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    infos: list[dict[str, Any]] = []
    for path in paths:
        path = _resolve_path(path)
        count = 0
        for line_no, row in _iter_jsonl(path):
            _check_unsafe(row, path, label="native role evidence")
            canonical_group_id = str(row.get("canonical_group_id") or "")
            if not canonical_group_id:
                raise ValueError(f"{path}:{line_no} missing canonical_group_id")
            source = str(row.get("maker_taker_attribution_source") or row.get("native_role_source") or "")
            if source not in ALLOWED_NATIVE_SOURCES:
                raise ValueError(f"{path}:{line_no} unsupported native role source {source!r}")
            if canonical_group_id in by_group:
                raise ValueError(f"{path}:{line_no} duplicate canonical_group_id={canonical_group_id}")
            by_group[canonical_group_id] = row
            count += 1
        infos.append({"path": str(path), "sha256": _sha256_file(path), "native_role_record_count": count})
    return by_group, infos


def _base_label(run_id: str, seq: int, timestamp_ns: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": "PHASE51N_MAKER_TAKER_ATTRIBUTION_RECOVERY_LABEL",
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


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        source = str(record.get("maker_taker_attribution_source") or "UNKNOWN")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def build_maker_taker_attribution_recovery(
    *,
    observed_pfill_run: Path,
    native_role_jsonl: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    run_id = run_id or f"PHASE51N-MAKER-TAKER-ATTRIBUTION-RECOVERY-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    observed_pfill_run = _resolve_path(observed_pfill_run)
    pfill_summary, pfill_labels = _load_observed_pfill(observed_pfill_run)
    native_by_group, native_inputs = _load_native_role_evidence(native_role_jsonl)

    records: list[dict[str, Any]] = []
    for seq, label in enumerate(pfill_labels, start=1):
        fill_count = _safe_int(label.get("fill_count")) or 0
        input_counts = _role_counts(label.get("maker_taker_role_counts"))
        effective_counts = dict(input_counts)
        canonical_group_id = str(label.get("canonical_group_id") or "")
        venue = str(label.get("venue_id") or "UNKNOWN")
        status = "NO_FILL_NOT_APPLICABLE"
        hold_reason = "no_fill"
        source = "NO_FILL_NOT_APPLICABLE"
        native_record = native_by_group.get(canonical_group_id)
        if fill_count > 0:
            if _known_count(input_counts) >= fill_count and int(input_counts.get("UNKNOWN") or 0) == 0:
                status = "OBSERVED_PRESERVED"
                hold_reason = "input_already_has_complete_maker_taker_roles"
                source = "INPUT_PFILL_ROLE_COUNTS"
            elif native_record is not None:
                native_counts = _role_counts(native_record.get("maker_taker_role_counts"))
                if _known_count(native_counts) >= fill_count and int(native_counts.get("UNKNOWN") or 0) == 0:
                    status = "RECOVERED_VENUE_NATIVE_ROLE"
                    hold_reason = "venue_native_role_evidence_supplied"
                    effective_counts = native_counts
                else:
                    status = "PARTIAL_VENUE_NATIVE_ROLE"
                    hold_reason = "native_role_evidence_incomplete_for_fill_count"
                    effective_counts = native_counts
                source = str(
                    native_record.get("maker_taker_attribution_source")
                    or native_record.get("native_role_source")
                    or "UNKNOWN"
                )
            else:
                status = "MISSING_VENUE_NATIVE_ROLE_SOURCE"
                hold_reason = "requires_venue_native_fill_or_trade_role_evidence"
                source = "UNKNOWN_NO_NATIVE_ROLE_SOURCE"
        record = _base_label(run_id, seq, timestamp_ns)
        record.update({
            "source": "phase51n_maker_taker_attribution_recovery",
            "observed_pfill_run": str(observed_pfill_run),
            "canonical_group_id": canonical_group_id,
            "canonical_order_key": label.get("order_key"),
            "source_telemetry_sha256": label.get("source_telemetry_sha256"),
            "venue_id": venue,
            "side": label.get("side"),
            "fill_count": fill_count,
            "input_maker_taker_role_counts": input_counts,
            "effective_maker_taker_role_counts": effective_counts,
            "maker_taker_recovery_status": status,
            "maker_taker_recovery_hold_reason": hold_reason,
            "maker_taker_attribution_source": source,
            "native_role_record_supplied": native_record is not None,
        })
        records.append(record)

    labels_path = out_dir / "maker_taker_attribution_recovery_labels.jsonl"
    summary_path = out_dir / "maker_taker_attribution_recovery_summary.json"
    _write_jsonl(labels_path, records)
    status_counts = _status_counts(records, "maker_taker_recovery_status")
    observed = status_counts.get("OBSERVED_PRESERVED", 0) + status_counts.get("RECOVERED_VENUE_NATIVE_ROLE", 0)
    incomplete = (
        status_counts.get("PARTIAL_VENUE_NATIVE_ROLE", 0)
        + status_counts.get("MISSING_VENUE_NATIVE_ROLE_SOURCE", 0)
    )
    gate_reason = (
        "phase51n_maker_taker_attribution_complete"
        if incomplete == 0
        else "phase51n_maker_taker_attribution_incomplete"
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
        "native_role_inputs": native_inputs,
        "label_count": len(records),
        "filled_count": sum(1 for record in records if int(record.get("fill_count") or 0) > 0),
        "maker_taker_observed_or_recovered_count": observed,
        "maker_taker_partial_or_missing_count": incomplete,
        "maker_taker_recovery_status_counts": status_counts,
        "maker_taker_attribution_source_counts": _source_counts(records),
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-pfill-run", type=Path, required=True)
    parser.add_argument("--native-role-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_maker_taker_attribution_recovery(
            observed_pfill_run=args.observed_pfill_run,
            native_role_jsonl=args.native_role_jsonl,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51n_maker_taker_attribution_recovery: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51n_maker_taker_attribution_recovery: wrote {out_dir}")
    print("phase51n_maker_taker_attribution_recovery: status HOLD (venue-native attribution evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
