#!/usr/bin/env python3
"""Build a Phase 5.1w forward capture request pack.

This HOLD-only gate converts a Phase 5.1u target manifest into an
operator-facing request pack for sanitized local source capture. It performs no
network access, reads no secrets, submits no orders, and does not validate or
infer venue truth. Phase 5.1v remains the readiness validator for any supplied
bundle.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51w_forward_capture_request_pack"
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
    "fill_id",
    "id",
    "oid",
    "cloid",
    "tid",
    "orderId",
    "clientOrderId",
    "tradeId",
    "fillId",
    "client_id",
    "clientId",
}
JOIN_PATHS = [
    "canonical_group_id",
    "order_key",
    "source-link sidecar keyed by redacted source-record hash",
]
PROHIBITED = [
    "live orders",
    "canary mode",
    "capital escalation",
    "risk-limit relaxation",
    "model training",
    "EV admission",
    "financial claims",
    "network source paths",
    ".env source files",
    "symlink source files",
    "secret-shaped fields",
    "raw venue identifiers",
    "maker/taker inference from intent",
    "Lighter limit inference from docs-only caps",
]


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _check_safe_record(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")
    raw_fields = RAW_IDENTIFIER_FIELDS & set(record)
    if raw_fields:
        raise ValueError(f"{path} has raw {label} identifier fields: {sorted(raw_fields)}")


def _load_target_run(target_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_path = target_run / "phase51u_forward_capture_target_manifest_summary.json"
    role_path = target_run / "native_role_capture_targets.jsonl"
    limit_path = target_run / "lighter_native_limit_capture_targets.jsonl"
    summary = _load_json(summary_path)
    _check_safe_record(summary, summary_path, label="target summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    role_targets = [row for _, row in _iter_jsonl(role_path)]
    limit_targets = [row for _, row in _iter_jsonl(limit_path)]
    for row in role_targets:
        _check_safe_record(row, role_path, label="role target")
    for row in limit_targets:
        _check_safe_record(row, limit_path, label="limit target")
    return summary, role_targets, limit_targets


def _counts_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _first_list(rows: list[dict[str, Any]], key: str) -> list[str]:
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _role_requirements(role_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    venues = sorted({str(row.get("venue_id") or "unknown") for row in role_targets})
    for venue in venues:
        rows = [row for row in role_targets if str(row.get("venue_id") or "unknown") == venue]
        source = str(rows[0].get("required_native_role_source") or f"{venue.upper()}_NATIVE_ROLE")
        out.append(
            {
                "source_id": f"{venue}_forward_native_role_snapshot",
                "venue_id": venue,
                "target_type": "native_role",
                "target_count": len(rows),
                "placeholder_path": f"<local-read-only-{venue}-native-role-snapshot.jsonl>",
                "required_source": source,
                "required_fields": _first_list(rows, "required_native_role_fields"),
                "accepted_join_paths": JOIN_PATHS,
                "public_or_shadow_data_sufficient": False,
                "source_link_only_sufficient": False,
            }
        )
    return out


def _limit_requirements(limit_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not limit_targets:
        return []
    return [
        {
            "source_id": "lighter_forward_native_limit_pressure_snapshot",
            "venue_id": "lighter",
            "target_type": "native_limit_pressure",
            "target_count": len(limit_targets),
            "placeholder_path": "<local-read-only-lighter-native-limit-pressure-snapshot.jsonl>",
            "required_source": "LIGHTER_LIMITS_AT_DECISION_TIME",
            "required_fields": _first_list(limit_targets, "required_native_limit_fields"),
            "accepted_native_limit_event_time_status": _first_list(
                limit_targets,
                "accepted_native_limit_event_time_status",
            ),
            "accepted_join_paths": JOIN_PATHS,
            "public_or_shadow_data_sufficient": False,
            "source_link_only_sufficient": False,
        }
    ]


def _capture_manifest_skeleton(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    sources = [
        {
            "source_id": req["source_id"],
            "venue_id": req["venue_id"],
            "path": req["placeholder_path"],
        }
        for req in requirements
    ]
    return {
        "manifest_version": 1,
        "baseline_commit": BASELINE_COMMIT,
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
        "sources": sources,
        "source_links": [
            {
                "source_link_id": "optional_phase51t_or_external_redacted_source_links",
                "path": "<local-redacted-source-links.sanitized.jsonl>",
            }
        ],
    }


def _markdown_pack(
    *,
    run_id: str,
    target_run: Path,
    role_targets: list[dict[str, Any]],
    limit_targets: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 5.1w Forward Capture Request Pack",
        "",
        f"Run id: `{run_id}`",
        "",
        "Status: `HOLD`. This request pack does not authorize live orders, canary, capital escalation, risk-limit relaxation, model training, EV admission, or financial claims.",
        "",
        "## Target Run",
        "",
        f"`{target_run}`",
        "",
        "## Required Targets",
        "",
        f"- Native role targets: `{len(role_targets)}`",
        f"- Lighter native-limit targets: `{len(limit_targets)}`",
        "",
        "Native role targets by venue:",
        "",
    ]
    for venue, count in _counts_by(role_targets, "venue_id").items():
        lines.append(f"- `{venue}`: `{count}`")
    lines.extend(
        [
            "",
            "## Required Local Files",
            "",
            "Each file must be local `.json` or `.jsonl`, sanitized, non-secret, non-symlink, and linked by `canonical_group_id`, `order_key`, or a redacted source-link sidecar.",
            "",
        ]
    )
    for req in requirements:
        lines.extend(
            [
                f"### `{req['source_id']}`",
                "",
                f"- Venue: `{req['venue_id']}`",
                f"- Target type: `{req['target_type']}`",
                f"- Target count: `{req['target_count']}`",
                f"- Placeholder path: `{req['placeholder_path']}`",
                f"- Required source: `{req['required_source']}`",
                "- Required fields:",
            ]
        )
        for field in req.get("required_fields", []):
            lines.append(f"  - `{field}`")
        if req.get("accepted_native_limit_event_time_status"):
            lines.append("- Accepted native-limit event-time statuses:")
            for status in req["accepted_native_limit_event_time_status"]:
                lines.append(f"  - `{status}`")
        lines.extend(
            [
                "- Public/shadow data sufficient: `false`",
                "- Source-link-only sufficient: `false`",
                "",
            ]
        )
    lines.extend(
        [
            "## Prohibited",
            "",
        ]
    )
    for item in PROHIBITED:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "After replacing placeholders in `capture_bundle_manifest.skeleton.json` with local sanitized paths:",
            "",
            "```bash",
            "python3 tools/phase51v_forward_capture_bundle_readiness.py \\",
            f"  --target-run {target_run} \\",
            "  --candidate-manifest <local_capture_bundle_manifest.json> \\",
            "  --run-id <phase51v_run_id>",
            "```",
            "",
            "Only if `generated_phase51s_manifest_ready=true`, run:",
            "",
            "```bash",
            "python3 tools/phase51s_local_native_source_acquisition.py \\",
            "  --manifest runs/phase51v_forward_capture_bundle_readiness/<phase51v_run_id>/phase51s_manifest.generated.json \\",
            "  --run-id <phase51s_run_id>",
            "```",
            "",
            "Then continue Phase 5.1s -> 5.1r -> 5.1q -> 5.1n -> 5.1h -> 5.1i.",
            "",
        ]
    )
    return "\n".join(lines)


def build_forward_capture_request_pack(
    *,
    target_run: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    target_run = _resolve_path(target_run)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    target_summary, role_targets, limit_targets = _load_target_run(target_run)
    requirements = _role_requirements(role_targets) + _limit_requirements(limit_targets)
    capture_manifest = _capture_manifest_skeleton(requirements)

    request_pack = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51w_forward_capture_request_pack_emitted_nonlive_hold",
        "target_run": str(target_run),
        "target_manifest_summary_sha256": _sha256_file(
            target_run / "phase51u_forward_capture_target_manifest_summary.json"
        ),
        "native_role_capture_target_count": len(role_targets),
        "native_role_capture_target_counts_by_venue": _counts_by(role_targets, "venue_id"),
        "native_role_required_source_counts": _counts_by(role_targets, "required_native_role_source"),
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "required_local_source_file_count": len(requirements),
        "required_source_files": requirements,
        "capture_bundle_manifest_skeleton_sha256": _stable_hash(capture_manifest),
        "phase51v_target_command": (
            "python3 tools/phase51v_forward_capture_bundle_readiness.py "
            f"--target-run {target_run} --candidate-manifest <local_capture_bundle_manifest.json> "
            "--run-id <phase51v_run_id>"
        ),
        "downstream_chain": "phase51v -> phase51s -> phase51r -> phase51q -> phase51n -> phase51h -> phase51i",
        "target_summary_gate_status": target_summary.get("gate_status"),
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
        "raw_identifier_redaction_status": "PASS",
        "prohibited": PROHIBITED,
    }

    request_json_path = out_dir / "forward_capture_request_pack.json"
    request_md_path = out_dir / "forward_capture_request_pack.md"
    capture_manifest_path = out_dir / "capture_bundle_manifest.skeleton.json"
    summary_path = out_dir / "phase51w_forward_capture_request_pack_summary.json"
    manifest_path = out_dir / "phase51w_manifest.json"

    _write_json(request_json_path, request_pack)
    _write_text(
        request_md_path,
        _markdown_pack(
            run_id=run_id,
            target_run=target_run,
            role_targets=role_targets,
            limit_targets=limit_targets,
            requirements=requirements,
        ),
    )
    _write_json(capture_manifest_path, capture_manifest)

    summary = {
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51w_forward_capture_request_pack_emitted_nonlive_hold",
        "target_run": str(target_run),
        "native_role_capture_target_count": len(role_targets),
        "native_role_capture_target_counts_by_venue": _counts_by(role_targets, "venue_id"),
        "lighter_native_limit_capture_target_count": len(limit_targets),
        "required_local_source_file_count": len(requirements),
        "request_pack_path": str(request_md_path),
        "request_pack_json_path": str(request_json_path),
        "capture_bundle_manifest_skeleton_path": str(capture_manifest_path),
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
        "raw_identifier_redaction_status": "PASS",
    }
    _write_json(summary_path, summary)

    artifacts = [request_json_path, request_md_path, capture_manifest_path, summary_path]
    manifest_out = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "artifacts": _artifact_infos(out_dir, artifacts),
    }
    _write_json(manifest_path, manifest_out)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51w_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_forward_capture_request_pack(
            target_run=args.target_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51w_forward_capture_request_pack: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51w_forward_capture_request_pack: wrote {out_dir}")
    print("phase51w_forward_capture_request_pack: status HOLD (request pack only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
