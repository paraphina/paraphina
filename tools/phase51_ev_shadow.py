#!/usr/bin/env python3
"""Phase 5.1 non-live EV shadow artifact generator.

This tool does not send orders and does not modify runtime services.  It reads
an optional v1 telemetry file, extracts Lighter quote-level candidates, and
emits schema_version=2 EV shadow records that intentionally default to HOLD
until Phase 5.1 calibration evidence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON in {path}")
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
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns() -> int:
    return time.time_ns()


def _resolve_output_dir(spec: dict[str, Any], output_root: Path | None, run_id: str) -> Path:
    root = output_root or ROOT / spec.get("output_root", "runs/phase51_lighter_only_ev_shadow")
    if not root.is_absolute():
        root = ROOT / root
    return root / run_id


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("spec baseline_commit does not match clean Phase 5 baseline")
    if spec.get("venue_id") != "lighter":
        raise ValueError("Phase 5.1 first experiment spec must be Lighter-only")
    if spec.get("no_live_flag") is not True:
        raise ValueError("no_live_flag must be true")
    if spec.get("capital_escalation_flag") is not False:
        raise ValueError("capital escalation must be false")
    if spec.get("risk_limit_override_flag") is not False:
        raise ValueError("risk limit override must be false")
    constraints = spec.get("constraints", {})
    if constraints.get("live_orders_allowed") is not False:
        raise ValueError("live_orders_allowed must be false")


def _base_event(
    *,
    event_type: str,
    event_seq: int,
    run_id: str,
    timestamp_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "event_type": event_type,
        "event_seq": event_seq,
        "timestamp_local_ns": timestamp_ns + event_seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
    }


def _iter_input_records(path: Path) -> tuple[int, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    scanned = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return scanned, records


def _quote_candidates_from_record(record: dict[str, Any], venue_id: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for quote in record.get("quote_levels", []) or []:
        if not isinstance(quote, dict):
            continue
        if quote.get("venue_id") != venue_id:
            continue
        side = quote.get("side")
        price = quote.get("price")
        size = quote.get("size_final")
        if not isinstance(price, (int, float)) or not isinstance(size, (int, float)):
            continue
        candidate_id = (
            f"{record.get('t', 'unknown')}:{venue_id}:{side}:"
            f"{quote.get('quote_state', 'unknown')}:{price}:{size}"
        )
        candidates.append({
            "candidate_id": candidate_id,
            "venue_id": venue_id,
            "side": str(side or "UNKNOWN").upper(),
            "layer": "BASELINE_COMPAT",
            "price": float(price),
            "size": max(float(size), 0.0),
            "local_edge_feature": float(quote.get("edge_local") or 0.0),
            "pair_edge_feature": None,
            "quote_state": quote.get("quote_state"),
            "decision_reason_primary": "missing_phase51_calibration",
        })
    return candidates


def _ev_event(candidate: dict[str, Any], *, event_seq: int, run_id: str, timestamp_ns: int, alpha: float) -> dict[str, Any]:
    event = _base_event(
        event_type="V2_EV_EVALUATED",
        event_seq=event_seq,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
    )
    event.update({
        "candidate_id": candidate["candidate_id"],
        "venue_id": candidate["venue_id"],
        "side": candidate["side"],
        "layer": candidate["layer"],
        "selected_size_Q": candidate["size"],
        "local_edge_feature": candidate["local_edge_feature"],
        "pair_edge_feature": candidate["pair_edge_feature"],
        "P_fill_hat": 0.0,
        "P_fill_ci_low": 0.0,
        "P_fill_ci_high": 0.0,
        "P_hedge_success_hat": 0.0,
        "P_hedge_partial_hat": 0.0,
        "P_hedge_fail_hat": 1.0,
        "E_locked_edge_hat": 0.0,
        "E_partial_hedge_state_hat": 0.0,
        "E_residual_inventory_state_hat": 0.0,
        "E_adverse_selection_hat": 0.0,
        "E_queue_reset_hat": 0.0,
        "E_churn_hat": 0.0,
        "E_capital_funding_hat": 0.0,
        "E_tail_risk_hat": 0.0,
        "EV_hat": 0.0,
        "EV_standard_error": 0.0,
        "EV_lcb_alpha": 0.0,
        "alpha": alpha,
        "decision": "HOLD",
        "decision_reason_primary": candidate["decision_reason_primary"],
        "admissible_for_financial_claim": False,
    })
    return event


def _write_evidence_index(out_dir: Path, artifact_paths: list[Path], metadata: dict[str, Any]) -> None:
    evidence_dir = out_dir / "evidence_pack"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "metadata": metadata,
        "artifacts": [
            {
                "path": path.relative_to(out_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifact_paths)
        ],
    }
    _write_json(evidence_dir / "artifact_index.json", index)
    try:
        manifest_module = ROOT / "batch_runs" / "evidence_pack" / "manifest.py"
        spec = importlib.util.spec_from_file_location("phase51_evidence_manifest", manifest_module)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load evidence manifest helper: {manifest_module}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        module.write_manifest(evidence_dir, metadata)
        manifest_error = evidence_dir / "manifest_error.json"
        if manifest_error.exists():
            manifest_error.unlink()
    except Exception as exc:  # pragma: no cover - manifest helper is optional here
        _write_json(evidence_dir / "manifest_error.json", {"error": str(exc)})


def run(spec_path: Path, output_root: Path | None, input_telemetry: Path | None, run_id: str | None) -> Path:
    spec = _load_json(spec_path)
    _validate_spec(spec)
    run_id = run_id or f"{spec.get('experiment_id', 'phase51_ev_shadow')}_{_utc_stamp()}"
    out_dir = _resolve_output_dir(spec, output_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = _timestamp_ns()
    alpha = float(spec.get("alpha", 0.05))

    resolved_spec = dict(spec)
    resolved_spec["run_id"] = run_id
    resolved_spec["spec_path"] = str(spec_path)
    resolved_spec["input_telemetry"] = str(input_telemetry) if input_telemetry else spec.get("input_telemetry")
    resolved_spec["output_dir"] = str(out_dir)

    events: list[dict[str, Any]] = [
        {
            **_base_event(
                event_type="V2_RUN_CONTEXT",
                event_seq=1,
                run_id=run_id,
                timestamp_ns=timestamp_ns,
            ),
            "venue_id": "lighter",
            "decision": "HOLD",
            "decision_reason_primary": "nonlive_ev_shadow_context",
            "admissible_for_financial_claim": False,
        }
    ]

    scanned_records = 0
    candidates = 0
    input_path = input_telemetry or (
        Path(spec["input_telemetry"]) if spec.get("input_telemetry") else None
    )
    if input_path:
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        scanned_records, records = _iter_input_records(input_path)
        for record in records:
            for candidate in _quote_candidates_from_record(record, "lighter"):
                candidates += 1
                events.append(_ev_event(
                    candidate,
                    event_seq=len(events) + 1,
                    run_id=run_id,
                    timestamp_ns=timestamp_ns,
                    alpha=alpha,
                ))
    else:
        events.append({
            **_base_event(
                event_type="V2_GUARDRAIL_EVENT",
                event_seq=2,
                run_id=run_id,
                timestamp_ns=timestamp_ns,
            ),
            "venue_id": "lighter",
            "decision": "HOLD",
            "decision_reason_primary": "no_input_telemetry",
            "admissible_for_financial_claim": False,
        })

    summary = {
        "experiment_id": spec.get("experiment_id"),
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "venue_id": "lighter",
        "input_records_scanned": scanned_records,
        "candidates_evaluated": candidates,
        "admit_count": 0,
        "reject_count": 0,
        "hold_count": candidates + (1 if not input_path else 0),
        "gate_status": "HOLD",
        "gate_reason": "nonlive_shadow_requires_calibration_and_board_review",
        "no_live_flag": True,
        "admissible_for_financial_claim": False,
    }
    gate = {
        "status": "HOLD",
        "reason": summary["gate_reason"],
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_nonlive_evidence_review": candidates > 0,
    }
    command_log = {
        "argv": sys.argv,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
    }

    artifact_paths = [
        out_dir / "spec_resolved.json",
        out_dir / "telemetry.jsonl",
        out_dir / "ev_shadow_summary.json",
        out_dir / "gate_result.json",
        out_dir / "command_log.json",
    ]
    _write_json(artifact_paths[0], resolved_spec)
    _write_jsonl(artifact_paths[1], events)
    _write_json(artifact_paths[2], summary)
    _write_json(artifact_paths[3], gate)
    _write_json(artifact_paths[4], command_log)
    if input_path:
        copied = out_dir / "input_telemetry.source.jsonl"
        shutil.copyfile(input_path, copied)
        artifact_paths.append(copied)

    _write_evidence_index(out_dir, artifact_paths, {
        "experiment_id": spec.get("experiment_id"),
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "no_live_flag": True,
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs/phase51_lighter_only_ev_shadow.json",
        help="Phase 5.1 EV shadow experiment spec JSON.",
    )
    parser.add_argument("--input-telemetry", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    try:
        out_dir = run(args.spec, args.output_root, args.input_telemetry, args.run_id)
    except Exception as exc:
        print(f"phase51_ev_shadow: ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"phase51_ev_shadow: wrote {out_dir}")
    print("phase51_ev_shadow: status HOLD (nonlive evidence only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
