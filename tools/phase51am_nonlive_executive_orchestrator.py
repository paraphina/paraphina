#!/usr/bin/env python3
"""Phase 5.1am non-live executive orchestrator.

This HOLD-only control tool turns the current Phase 5.1 blocker state into an
auditable autonomous work queue. It does not mine new venue evidence, does not
touch credentials, does not call network endpoints, does not place/cancel
orders, and does not infer source links or Lighter pressure. When no admissible
repo-owned route can run, it emits machine-readable work packets and a
source-owner request instead of leaving the workflow at an unstructured stop.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51am_nonlive_executive_orchestrator"

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
    "auth_token",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "session_token",
    "signing_key",
    "token",
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

ROUTE_PRIORITY = [
    "forward_refresh",
    "validated_mapping",
    "direct_private_rows",
    "lighter_pressure",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _check_unsafe_flags(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_no_secret_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_no_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        raw_fields = sorted(RAW_IDENTIFIER_FIELDS & set(obj))
        if raw_fields:
            raise ValueError(f"{path} {label} leaked raw identifier fields: {raw_fields}")


def _check_safety(record: Any, path: Path, *, label: str) -> None:
    _check_unsafe_flags(record, path, label=label)
    _check_no_secret_fields(record, path, label=label)
    _check_no_raw_identifier_fields(record, path, label=label)


def _path_text_is_unsafe(path_text: str) -> bool:
    return "://" in path_text or path_text.lower().startswith(("http:", "https:", "s3:", "gs:"))


def _path_contains_env_part(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _path_has_symlink(path: Path) -> bool:
    probe = path
    candidates = [probe, *probe.parents]
    for candidate in candidates:
        if candidate.exists() and candidate.is_symlink():
            return True
        if candidate == candidate.parent:
            break
    return False


def _check_local_path(path: Path, repo_root: Path, *, label: str, must_exist: bool = True) -> Path:
    path_text = str(path)
    if _path_text_is_unsafe(path_text):
        raise ValueError(f"{label} must be a local filesystem path, got {path_text!r}")
    resolved = _resolve_path(path, repo_root)
    if _path_contains_env_part(resolved):
        raise ValueError(f"{label} must not reference .env files: {resolved}")
    if must_exist and not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if resolved.exists() and _path_has_symlink(resolved):
        raise ValueError(f"{label} must not traverse symlinks: {resolved}")
    return resolved


def _load_json(path: Path, repo_root: Path, *, label: str) -> Any:
    checked = _check_local_path(path, repo_root, label=label)
    with checked.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    _check_safety(payload, checked, label=label)
    return payload


def _iter_json_or_jsonl_records(path: Path, repo_root: Path, *, label: str) -> tuple[int, str]:
    checked = _check_local_path(path, repo_root, label=label)
    suffix = checked.suffix.lower()
    if suffix == ".json":
        with checked.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        _check_safety(payload, checked, label=label)
        if isinstance(payload, list):
            return len(payload), "json_list"
        if isinstance(payload, dict):
            for key in ("rows", "data", "result", "items", "mappings"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value), f"json_object_{key}"
            return 1, "json_object"
        raise ValueError(f"{checked} {label} must contain a JSON object or list")
    row_count = 0
    with checked.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{checked} {label} expected object at line {line_no}")
            _check_safety(row, checked, label=label)
            row_count += 1
    return row_count, "jsonl"


def _checked_artifact_record(path: Path, repo_root: Path, *, label: str) -> dict[str, Any]:
    checked = _check_local_path(path, repo_root, label=label)
    row_count, artifact_format = _iter_json_or_jsonl_records(checked, repo_root, label=label)
    return {
        "path": str(checked),
        "sha256": _sha256_file(checked),
        "row_count": row_count,
        "artifact_format": artifact_format,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_safety(data, path, label="output")
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            _check_safety(record, path, label="output")
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_filename(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    name = "".join(chars).strip("_")
    return name or "packet"


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    infos = []
    for path in artifact_paths:
        try:
            display_path = str(path.relative_to(root_dir))
        except ValueError:
            display_path = str(path)
        infos.append(
            {
                "path": display_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return infos


def _parse_generated_at(value: Any) -> float:
    if not value:
        return 0.0
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _summary_sort_key(record: dict[str, Any]) -> tuple[float, str]:
    path = Path(str(record["path"]))
    return (_parse_generated_at(record["summary"].get("generated_at_utc")), str(path))


def _collect_json_summaries(
    repo_root: Path,
    explicit_paths: list[Path],
    default_glob: str,
    *,
    label: str,
) -> list[dict[str, Any]]:
    paths = [_check_local_path(path, repo_root, label=label) for path in explicit_paths]
    if not paths:
        paths = sorted(repo_root.glob(default_glob))
    records = []
    for path in paths:
        if not path.exists():
            continue
        summary = _load_json(path, repo_root, label=label)
        if not isinstance(summary, dict):
            raise ValueError(f"{path} must contain a JSON object")
        records.append(
            {
                "path": str(path),
                "summary": summary,
                "sha256": _sha256_file(path),
            }
        )
    records.sort(key=_summary_sort_key, reverse=True)
    return records


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _discover_previous_phase51am_summary(
    repo_root: Path,
    out_dir: Path,
    explicit_summary: Path | None,
) -> dict[str, Any] | None:
    explicit_paths = [explicit_summary] if explicit_summary is not None else []
    records = _collect_json_summaries(
        repo_root,
        explicit_paths,
        "runs/phase51am_nonlive_executive_orchestrator/*/phase51am_nonlive_executive_orchestrator_summary.json",
        label="previous Phase 5.1am summary",
    )
    for record in records:
        if not _is_relative_to(Path(str(record["path"])), out_dir):
            return record
    return None


def _as_list(value: Any, *, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"source-owner intake field {field} must be a list")
    return value


def _path_list_from_manifest(payload: dict[str, Any], field: str) -> list[Path]:
    paths = []
    for index, value in enumerate(_as_list(payload.get(field), field=field)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"source-owner intake field {field}[{index}] must be a non-empty string")
        paths.append(Path(value))
    return paths


def _string_list_from_manifest(payload: dict[str, Any], field: str) -> list[str]:
    values = []
    for index, value in enumerate(_as_list(payload.get(field), field=field)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"source-owner intake field {field}[{index}] must be a non-empty string")
        values.append(value)
    return values


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    deduped = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _load_source_owner_intake_manifest(repo_root: Path, manifest_path: Path | None) -> dict[str, Any]:
    empty = {
        "manifest_path": None,
        "manifest_sha256": None,
        "material_change_reason": None,
        "phase51al_summaries": [],
        "validated_mappings": [],
        "phase51aj_source_json": [],
        "phase51ab_pressure_jsonls": [],
    }
    if manifest_path is None:
        return empty
    checked_path = _check_local_path(manifest_path, repo_root, label="source-owner intake manifest")
    payload = _load_json(checked_path, repo_root, label="source-owner intake manifest")
    if not isinstance(payload, dict):
        raise ValueError(f"{checked_path} must contain a JSON object")
    allowed_keys = {
        "schema_version",
        "material_change_reason",
        "phase51al_summaries",
        "validated_mappings",
        "phase51aj_source_json",
        "phase51ab_pressure_jsonls",
        "no_live_flag",
        "approved_for_live",
        "approved_for_canary",
        "approved_for_model_training",
        "approved_for_capital_escalation",
        "admissible_for_financial_claim",
        "admissible_for_ev_admission",
        "live_orders_allowed",
        "capital_change_allowed",
        "risk_limit_relaxation_allowed",
    }
    unexpected = sorted(set(payload) - allowed_keys)
    if unexpected:
        raise ValueError(f"{checked_path} has unexpected source-owner intake fields: {unexpected}")
    return {
        "manifest_path": str(checked_path),
        "manifest_sha256": _sha256_file(checked_path),
        "material_change_reason": payload.get("material_change_reason"),
        "phase51al_summaries": _path_list_from_manifest(payload, "phase51al_summaries"),
        "validated_mappings": _path_list_from_manifest(payload, "validated_mappings"),
        "phase51aj_source_json": _string_list_from_manifest(payload, "phase51aj_source_json"),
        "phase51ab_pressure_jsonls": _path_list_from_manifest(payload, "phase51ab_pressure_jsonls"),
    }


def _source_owner_intake_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "material_change_reason": "describe_the_new_source_owner_truth_or_material_change",
        "phase51al_summaries": [],
        "validated_mappings": [],
        "phase51aj_source_json": [],
        "phase51ab_pressure_jsonls": [],
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
    }


def _source_owner_intake_status(
    intake: dict[str, Any],
    phase51al_summaries: list[Path],
    validated_mappings: list[Path],
    phase51aj_source_specs: list[str],
    phase51ab_pressure_jsonls: list[Path],
) -> dict[str, Any]:
    return {
        "manifest_supplied": intake.get("manifest_path") is not None,
        "manifest_path": intake.get("manifest_path"),
        "manifest_sha256": intake.get("manifest_sha256"),
        "material_change_reason_supplied": bool(intake.get("material_change_reason")),
        "phase51al_summary_count": len(phase51al_summaries),
        "validated_mapping_count": len(validated_mappings),
        "phase51aj_source_spec_count": len(phase51aj_source_specs),
        "phase51ab_pressure_jsonl_count": len(phase51ab_pressure_jsonls),
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
    }


def _render_subagent_prompt(packet: dict[str, Any]) -> str:
    lines = [
        f"# {packet.get('packet', 'subagent_packet')}",
        "",
        f"Lane: {packet.get('lane', 'unspecified')}",
        f"Status: {packet.get('status', 'READY_TO_DISPATCH')}",
        f"Priority: {packet.get('priority', 'unspecified')}",
        "",
        "## Mission",
        "",
        str(packet.get("mission", "")),
        "",
        "## Prompt",
        "",
        str(packet.get("prompt", "")),
        "",
    ]
    if packet.get("command_template"):
        lines.extend(["## Command Template", "", "```text", str(packet["command_template"]), "```", ""])
    for field, title in (
        ("required_artifacts", "Required Artifacts"),
        ("acceptance_checks", "Acceptance Checks"),
        ("stop_conditions", "Stop Conditions"),
    ):
        values = packet.get(field) or []
        if values:
            lines.extend([f"## {title}", ""])
            for value in values:
                lines.append(f"- {value}")
            lines.append("")
    lines.extend(
        [
            "## Return Format",
            "",
            "Return a concise verdict, changed paths if any, evidence paths/counts, safety issues, and the next safe action.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_subagent_prompt_pack(out_dir: Path, work_packets: list[dict[str, Any]]) -> tuple[Path, Path, list[Path]]:
    prompt_dir = out_dir / "subagent_prompt_pack"
    prompt_paths = []
    index_records = []
    for offset, packet in enumerate(work_packets, start=1):
        priority = int(packet.get("priority") or offset)
        packet_name = str(packet.get("packet") or f"packet_{offset}")
        path = prompt_dir / f"{priority:02d}_{_safe_filename(packet_name)}.md"
        _write_text(path, _render_subagent_prompt(packet))
        prompt_paths.append(path)
        index_records.append(
            {
                "packet": packet_name,
                "priority": priority,
                "lane": packet.get("lane"),
                "status": packet.get("status"),
                "prompt_path": str(path),
                "prompt_sha256": _sha256_file(path),
            }
        )
    index_path = prompt_dir / "index.json"
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "prompt_count": len(prompt_paths),
            "prompts": index_records,
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        },
    )
    return prompt_dir, index_path, prompt_paths


def _count_nonempty_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _extract_current_blocker(phase51ak_record: dict[str, Any] | None) -> dict[str, Any]:
    if phase51ak_record is None:
        return {
            "phase51ak_summary_path": None,
            "native_role_missing_by_venue": {},
            "status": "missing_phase51ak_summary",
        }

    summary = phase51ak_record["summary"]
    status_counts = summary.get("decision_status_counts_by_target_type_venue") or {}
    missing_by_venue: dict[str, int] = {}
    if isinstance(status_counts, dict):
        for key, value in status_counts.items():
            parts = str(key).split(":")
            if len(parts) != 3:
                continue
            target_type, venue, status = parts
            if target_type == "native_role" and status == "UNRECOVERABLE_FROM_LOCAL_ARTIFACTS":
                missing_by_venue[venue] = int(value or 0)

    decision_status_counts = summary.get("decision_status_counts") or {}
    unrecoverable_count = None
    if isinstance(decision_status_counts, dict):
        unrecoverable_count = decision_status_counts.get("UNRECOVERABLE_FROM_LOCAL_ARTIFACTS")

    return {
        "phase51ak_summary_path": phase51ak_record["path"],
        "phase51ak_summary_sha256": phase51ak_record["sha256"],
        "phase51ak_run_id": summary.get("run_id"),
        "target_pack_mode": summary.get("target_pack_mode"),
        "gate_status": summary.get("gate_status"),
        "gate_reason": summary.get("gate_reason"),
        "native_role_capture_target_count": summary.get("native_role_capture_target_count"),
        "native_role_capture_target_ready_count": summary.get("native_role_capture_target_ready_count"),
        "native_role_capture_target_missing_count": summary.get("native_role_capture_target_missing_count"),
        "source_owner_native_role_evidence_ready": summary.get("source_owner_native_role_evidence_ready"),
        "source_owner_native_role_ready_without_h_i": summary.get("source_owner_native_role_ready_without_h_i"),
        "phase51_source_owner_blocker_status": summary.get("phase51_source_owner_blocker_status"),
        "lighter_pressure_unavailable_governance_accepted": summary.get(
            "lighter_pressure_unavailable_governance_accepted"
        ),
        "h_i_feature_matrix_deferred": summary.get("h_i_feature_matrix_deferred"),
        "h_i_feature_matrix_deferred_reason": summary.get("h_i_feature_matrix_deferred_reason"),
        "source_owner_scope_next_required_action": summary.get("source_owner_scope_next_required_action"),
        "native_role_missing_by_venue": missing_by_venue,
        "lighter_native_limit_capture_target_count": summary.get("lighter_native_limit_capture_target_count"),
        "lighter_native_limit_capture_target_ready_count": summary.get("lighter_native_limit_capture_target_ready_count"),
        "lighter_native_limit_capture_target_missing_count": summary.get("lighter_native_limit_capture_target_missing_count"),
        "phase51v_downstream_chain_ready": summary.get("phase51v_downstream_chain_ready"),
        "decision_status_counts": decision_status_counts,
        "unrecoverable_from_local_artifacts_count": unrecoverable_count,
        "target_run": summary.get("target_run"),
        "request_pack": summary.get("request_pack"),
        "next_required_action": summary.get("next_required_action"),
        "status": "loaded",
    }


def _command_path(path_text: Any) -> str:
    return str(path_text) if path_text is not None else "<missing>"


def _forward_refresh_command(summary: dict[str, Any]) -> str:
    existing = summary.get("phase51ak_forward_refresh_command")
    if existing:
        return str(existing)
    return (
        "python3 tools/phase51ak_blocker_resolution_runner.py "
        f"--target-run {_command_path(summary.get('target_run'))} "
        f"--request-pack {_command_path(summary.get('request_pack'))} "
        "--no-default-current-manifest "
        f"--candidate-manifest {_command_path(summary.get('candidate_manifest_path'))} "
        "--target-pack-mode forward-refresh"
    )


def _classify_forward_refresh(
    repo_root: Path,
    phase51al_records: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    ready_candidates = []
    fixture_count = 0
    for record in phase51al_records:
        summary = record["summary"]
        run_id = str(summary.get("run_id") or Path(record["path"]).parent.name)
        is_fixture = "FIXTURE" in run_id.upper()
        fixture_count += 1 if is_fixture else 0
        required_paths = {
            "target_run": summary.get("target_run"),
            "request_pack": summary.get("request_pack"),
            "candidate_manifest_path": summary.get("candidate_manifest_path"),
        }
        missing_paths = []
        for key, path_text in required_paths.items():
            if not path_text or not _resolve_path(Path(str(path_text)), repo_root).exists():
                missing_paths.append(key)
        source_row_count = int(summary.get("source_row_count") or 0)
        target_count = int(summary.get("native_role_capture_target_count") or 0) + int(
            summary.get("lighter_native_limit_capture_target_count") or 0
        )
        candidate = {
            "run_id": run_id,
            "summary_path": record["path"],
            "summary_sha256": record["sha256"],
            "is_fixture": is_fixture,
            "source_row_count": source_row_count,
            "target_count": target_count,
            "missing_required_artifact_keys": missing_paths,
            "command_template": _forward_refresh_command(summary),
        }
        if not is_fixture and not missing_paths and source_row_count > 0 and target_count > 0:
            candidate["candidate_status"] = "READY_TO_VALIDATE"
            ready_candidates.append(candidate)
        elif is_fixture:
            candidate["candidate_status"] = "FIXTURE_ONLY_NONBLOCKING"
        elif missing_paths:
            candidate["candidate_status"] = "MISSING_REQUIRED_ARTIFACTS"
        else:
            candidate["candidate_status"] = "INCOMPLETE_FORWARD_REFRESH_INPUT"
        candidates.append(candidate)

    if ready_candidates:
        selected = ready_candidates[0]
        return {
            "route": "forward_refresh",
            "route_status": "READY_TO_VALIDATE",
            "rationale": "real Phase 5.1al forward-refresh pack is present and can be validated through Phase 5.1ak",
            "next_required_action": "run_phase51ak_with_target_pack_mode_forward_refresh",
            "command_template": selected["command_template"],
            "selected_candidate": selected,
            "candidate_count": len(candidates),
            "ready_candidate_count": len(ready_candidates),
            "fixture_candidate_count": fixture_count,
            "candidates": candidates,
        }

    rationale = "no Phase 5.1al summaries found"
    if candidates and fixture_count == len(candidates):
        rationale = "only deterministic Phase 5.1al fixture packs are present"
    elif candidates:
        rationale = "no real Phase 5.1al pack has complete required artifacts"
    return {
        "route": "forward_refresh",
        "route_status": "BLOCKED",
        "blocked_reason": "missing_real_forward_refresh_pack",
        "rationale": rationale,
        "next_required_action": "obtain_real_phase51al_forward_refresh_input",
        "candidate_count": len(candidates),
        "ready_candidate_count": 0,
        "fixture_candidate_count": fixture_count,
        "candidates": candidates,
    }


def _classify_validated_mapping(
    repo_root: Path,
    mapping_paths: list[Path],
    current_blocker: dict[str, Any],
) -> dict[str, Any]:
    request_pack = current_blocker.get("request_pack")
    if not mapping_paths:
        return {
            "route": "validated_mapping",
            "route_status": "BLOCKED",
            "blocked_reason": "missing_validated_redacted_mapping",
            "rationale": "no validated redacted source-link mapping was supplied",
            "next_required_action": "obtain_phase51ad_mapping",
            "mapping_count": 0,
        }
    checked_records = [
        _checked_artifact_record(path, repo_root, label="validated mapping")
        for path in mapping_paths
    ]
    if not request_pack:
        return {
            "route": "validated_mapping",
            "route_status": "BLOCKED",
            "blocked_reason": "missing_request_pack_from_phase51ak",
            "rationale": "Phase 5.1ad needs the current request pack from Phase 5.1ak",
            "next_required_action": "rerun_phase51ak_or_supply_request_pack",
            "mapping_count": len(checked_records),
        }
    command = (
        "python3 tools/phase51ad_source_link_sidecar_materialize.py "
        f"--request-pack {request_pack} "
        f"--mapping {checked_records[0]['path']}"
    )
    return {
        "route": "validated_mapping",
        "route_status": "READY_TO_STAGE",
        "rationale": "validated mapping path was supplied; Phase 5.1ad can materialize the source-link sidecar",
        "next_required_action": "run_phase51ad_then_phase51ak",
        "command_template": command,
        "mapping_count": len(checked_records),
        "mapping_paths": [record["path"] for record in checked_records],
        "mapping_records": checked_records,
    }


def _parse_source_spec(spec: str, repo_root: Path) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"source spec must be venue=/path/to/file, got {spec!r}")
    venue, raw_path = spec.split("=", 1)
    if not venue:
        raise ValueError(f"source spec missing venue: {spec!r}")
    path = Path(_checked_artifact_record(Path(raw_path), repo_root, label=f"{venue} direct private source")["path"])
    return venue, path


def _classify_direct_private_rows(
    repo_root: Path,
    source_specs: list[str],
    current_blocker: dict[str, Any],
) -> dict[str, Any]:
    request_pack = current_blocker.get("request_pack")
    if not source_specs:
        return {
            "route": "direct_private_rows",
            "route_status": "BLOCKED",
            "blocked_reason": "missing_direct_private_native_rows",
            "rationale": "no materially new directly target-linkable private/native rows were supplied",
            "next_required_action": "obtain_phase51aj_direct_rows",
            "source_spec_count": 0,
        }
    parsed = [_parse_source_spec(spec, repo_root) for spec in source_specs]
    if not request_pack:
        return {
            "route": "direct_private_rows",
            "route_status": "BLOCKED",
            "blocked_reason": "missing_request_pack_from_phase51ak",
            "rationale": "Phase 5.1aj needs the current request pack from Phase 5.1ak",
            "next_required_action": "rerun_phase51ak_or_supply_request_pack",
            "source_spec_count": len(parsed),
        }
    source_args = " ".join(f"--source-json {venue}={path}" for venue, path in parsed)
    command = (
        "python3 tools/phase51aj_forward_private_stream_source.py "
        f"--request-pack {request_pack} "
        f"{source_args}"
    )
    return {
        "route": "direct_private_rows",
        "route_status": "READY_TO_STAGE",
        "rationale": "direct private/native row paths were supplied; Phase 5.1aj can normalize them",
        "next_required_action": "run_phase51aj_then_phase51ak",
        "command_template": command,
        "source_spec_count": len(parsed),
        "source_specs": [f"{venue}={path}" for venue, path in parsed],
    }


def _discover_lighter_pressure(repo_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(repo_root.glob("runs/**/lighter_forward_native_limit_pressure_snapshot.jsonl")):
        if not path.exists() or path.is_dir():
            continue
        row_count = _count_nonempty_jsonl(path)
        records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "row_count": row_count,
            }
        )
    records.sort(key=lambda record: (record["row_count"] > 0, record["path"]), reverse=True)
    return records


def _classify_lighter_pressure(
    repo_root: Path,
    pressure_paths: list[Path],
    current_blocker: dict[str, Any],
) -> dict[str, Any]:
    target_run = current_blocker.get("target_run")
    explicit_records = []
    for path in pressure_paths:
        checked_record = _checked_artifact_record(path, repo_root, label="Lighter pressure source")
        explicit_records.append(
            {
                "path": checked_record["path"],
                "sha256": checked_record["sha256"],
                "row_count": checked_record["row_count"],
                "artifact_format": checked_record["artifact_format"],
            }
        )
    discovered_records = _discover_lighter_pressure(repo_root)
    candidate_records = explicit_records or [record for record in discovered_records if int(record["row_count"]) > 0]
    nonempty = [record for record in candidate_records if int(record["row_count"]) > 0]

    if nonempty and target_run:
        selected = nonempty[0]
        command = (
            "python3 tools/phase51ab_lighter_native_limit_pressure_source.py "
            f"--target-run {target_run} "
            f"--input-jsonl {selected['path']}"
        )
        return {
            "route": "lighter_pressure",
            "route_status": "READY_TO_STAGE",
            "rationale": "complete candidate Lighter pressure rows are present for Phase 5.1ab preflight",
            "next_required_action": "run_phase51ab_then_phase51ak",
            "command_template": command,
            "selected_pressure_path": selected["path"],
            "selected_pressure_row_count": selected["row_count"],
            "explicit_pressure_count": len(explicit_records),
            "discovered_pressure_count": len(discovered_records),
            "discovered_nonempty_pressure_count": len([record for record in discovered_records if int(record["row_count"]) > 0]),
        }

    blocked_reason = "missing_complete_lighter_pressure_rows"
    rationale = "no complete Lighter event-time pressure rows were supplied or discovered"
    if explicit_records and not nonempty:
        blocked_reason = "empty_lighter_pressure_rows"
        rationale = "supplied Lighter pressure files contain zero non-empty rows"
    elif nonempty and not target_run:
        blocked_reason = "missing_target_run_from_phase51ak"
        rationale = "Phase 5.1ab needs a target run from Phase 5.1ak"
    return {
        "route": "lighter_pressure",
        "route_status": "BLOCKED",
        "blocked_reason": blocked_reason,
        "rationale": rationale,
        "next_required_action": "obtain_phase51ab_complete_lighter_pressure_rows",
        "explicit_pressure_count": len(explicit_records),
        "discovered_pressure_count": len(discovered_records),
        "discovered_nonempty_pressure_count": len([record for record in discovered_records if int(record["row_count"]) > 0]),
    }


def _select_route(decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
    ready = [decision for decision in decisions if str(decision.get("route_status", "")).startswith("READY")]
    if not ready:
        return None
    ready_by_route = {decision["route"]: decision for decision in ready}
    for route in ROUTE_PRIORITY:
        if route in ready_by_route:
            return ready_by_route[route]
    return ready[0]


def _current_counts_text(current_blocker: dict[str, Any]) -> str:
    native_ready = current_blocker.get("native_role_capture_target_ready_count", "unknown")
    native_count = current_blocker.get("native_role_capture_target_count", "unknown")
    native_missing = current_blocker.get("native_role_capture_target_missing_count", "unknown")
    limit_ready = current_blocker.get("lighter_native_limit_capture_target_ready_count", "unknown")
    limit_count = current_blocker.get("lighter_native_limit_capture_target_count", "unknown")
    return (
        f"Current Phase 5.1ak state: native-role ready {native_ready} / {native_count}, "
        f"native-role missing {native_missing}, Lighter pressure ready {limit_ready} / {limit_count}."
    )


def _build_no_route_work_packets(current_blocker: dict[str, Any]) -> list[dict[str, Any]]:
    counts_text = _current_counts_text(current_blocker)
    return [
        {
            "packet": "phase51am_forward_refresh_source_owner",
            "lane": "Execution/Venue Lead",
            "priority": 1,
            "status": "READY_TO_DISPATCH",
            "mission": "Obtain a real sanitized Phase 5.1al forward-refresh input where target keys and native source truth are captured together at event time.",
            "prompt": (
                f"{counts_text} Produce or locate a non-live source-owner forward-refresh bundle. "
                "It must contain deterministic target keys plus venue-native role fields, or complete "
                "Lighter event-time pressure fields, and must be stageable through "
                "tools/phase51al_forward_refresh_capture_gate.py."
            ),
            "required_artifacts": [
                "sanitized Phase 5.1al input JSONL",
                "no secrets or raw private identifiers",
                "target keys captured with source truth at event time",
            ],
            "acceptance_checks": [
                "phase51al summary is not a fixture",
                "Phase 5.1ak validates it with --target-pack-mode forward-refresh",
                "all live/canary/capital/risk flags remain false",
            ],
            "stop_conditions": [
                "requires exposing credentials or raw private identifiers",
                "requires live orders, cancels, replacements, transfers, or account mutation",
            ],
        },
        {
            "packet": "phase51am_validated_mapping_source_owner",
            "lane": "Systems/Data Lead",
            "priority": 2,
            "status": "READY_TO_DISPATCH",
            "mission": "Obtain a validated redacted source-link mapping for the current request pack.",
            "prompt": (
                f"{counts_text} Find or request a redacted mapping from source_record_sha256 to "
                "canonical_group_id or order_key for the current Phase 5.1z request pack. "
                "Do not infer links from time, price, size, venue role, or proximity."
            ),
            "required_artifacts": [
                "validated mapping accepted by tools/phase51ad_source_link_sidecar_materialize.py",
                "cross-venue joins rejected",
                "unknown source hashes rejected",
            ],
            "acceptance_checks": [
                "Phase 5.1ad materializes source_links.sanitized.jsonl",
                "Phase 5.1ak remains the final wrapper",
            ],
            "stop_conditions": [
                "mapping contains raw order, trade, client, transaction, or account identifiers",
                "mapping depends on probabilistic or proximity matching",
            ],
        },
        {
            "packet": "phase51am_direct_private_rows_source_owner",
            "lane": "Execution/Venue Lead",
            "priority": 3,
            "status": "READY_TO_DISPATCH",
            "mission": "Obtain materially new directly target-linkable private/native rows for Phase 5.1aj.",
            "prompt": (
                f"{counts_text} Locate sanitized private/native rows where the same local row carries "
                "venue-native role truth and deterministic target linkage. Rows that only duplicate "
                "already-ready Extended or Paradex targets do not advance the blocker."
            ),
            "required_artifacts": [
                "local JSON, JSONL, or NDJSON source rows in venue=path form",
                "venue-native role field present in each useful row",
                "deterministic target linkage present without inference",
            ],
            "acceptance_checks": [
                "Phase 5.1aj emits new target-linked source rows",
                "Phase 5.1ak decision counts improve without unsafe flags",
            ],
            "stop_conditions": [
                "row requires committing raw private identifiers",
                "row links only by time, price, size, account role, or proximity",
            ],
        },
        {
            "packet": "phase51am_lighter_pressure_source_owner",
            "lane": "Execution/Venue Lead",
            "priority": 4,
            "status": "READY_TO_DISPATCH",
            "mission": "Obtain complete sanitized Lighter event-time native-limit pressure rows for Phase 5.1ab.",
            "prompt": (
                f"{counts_text} Produce event-time Lighter pressure rows with active-order headroom, "
                "sendTx limit and remaining, REST-or-weighted limit and remaining, and explicit "
                "event-time alignment. Documentation-only caps, account tiers, current snapshots, "
                "empty headers, and config strings are not admissible."
            ),
            "required_artifacts": [
                "lighter_forward_native_limit_pressure_snapshot-compatible JSONL",
                "native_limit_event_time_status=EVENT_TIME_ALIGNED",
                "complete sendTx and REST-or-weighted pressure dimensions",
            ],
            "acceptance_checks": [
                "Phase 5.1ab accepts the rows",
                "Phase 5.1ak final validation recognizes improved Lighter pressure readiness",
            ],
            "stop_conditions": [
                "requires sendTx or any venue write path",
                "pressure fields are documentation-only or current-snapshot-only",
            ],
        },
        {
            "packet": "phase51am_independent_audit",
            "lane": "Independent Auditor",
            "priority": 5,
            "status": "READY_TO_DISPATCH",
            "mission": "Audit whether the emitted work packets remain the highest-leverage safe continuation.",
            "prompt": (
                "Review the Phase 5.1am summary, decision ledger, current Phase 5.1ak blocker state, "
                "and generated source-owner request. Confirm the workflow advances autonomously without "
                "restarting exhausted retrospective mining or weakening evidence rules."
            ),
            "required_artifacts": [
                "challenge memo with HOLD/PROMOTE/ROLLBACK recommendation",
                "explicit answer to whether this remains the highest-leverage safe move",
            ],
            "acceptance_checks": [
                "no source-link or pressure inference is introduced",
                "no live/canary/capital/risk authorization is implied",
            ],
            "stop_conditions": [
                "source-of-truth docs contradict the emitted route ledger",
                "workflow would require secrets or live account mutation",
            ],
        },
    ]


def _build_ready_route_packets(selected_route: dict[str, Any], current_blocker: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "packet": "phase51am_execute_selected_route",
            "lane": "Systems Implementer",
            "priority": 1,
            "status": "READY_TO_DISPATCH",
            "mission": f"Run the selected non-live route: {selected_route['route']}.",
            "prompt": (
                f"{_current_counts_text(current_blocker)} Execute the selected command template in a "
                "bounded non-live workspace, then rerun Phase 5.1ak as the final wrapper where applicable."
            ),
            "command_template": selected_route.get("command_template"),
            "acceptance_checks": [
                "generated artifacts retain gate_status HOLD",
                "all live/canary/capital/risk flags remain false",
                "raw identifier redaction status is PASS",
            ],
            "stop_conditions": [
                "command requests network, credentials, live orders, cancels, or account mutation",
                "artifact includes unsafe true flags, secret-shaped fields, or raw identifier fields",
            ],
        },
        {
            "packet": "phase51am_ready_route_audit",
            "lane": "Independent Auditor",
            "priority": 2,
            "status": "READY_TO_DISPATCH",
            "mission": "Audit the selected route before any promotion claim.",
            "prompt": (
                "Confirm that the selected route is evidence-gated, non-live, and materially advances "
                "the blocker before it is used to continue the Phase 5.1 ladder."
            ),
            "acceptance_checks": [
                "route selection follows Phase 5.1am priority order",
                "Phase 5.1ak remains the final blocker-resolution wrapper",
            ],
            "stop_conditions": [
                "selected artifacts are fixture-only",
                "selected artifacts do not contain new source-owner truth",
            ],
        },
    ]


def _build_source_owner_request(current_blocker: dict[str, Any], selected_route: dict[str, Any] | None) -> str:
    missing_by_venue = current_blocker.get("native_role_missing_by_venue") or {}
    if selected_route is not None:
        return "\n".join(
            [
                "# Phase 5.1am Source-Owner Request",
                "",
                "A non-live route is already ready to stage or validate.",
                "",
                f"Selected route: `{selected_route['route']}`.",
                f"Next action: `{selected_route.get('next_required_action')}`.",
                "",
                "Do not add new source-owner material unless the selected route fails its safety or readiness checks.",
                "",
            ]
        )

    lines = [
        "# Phase 5.1am Source-Owner Request",
        "",
        "Current local/read-only artifacts do not contain an admissible blocker-clearing route.",
        "",
        _current_counts_text(current_blocker),
        "",
        "Native-role missing by venue:",
    ]
    if missing_by_venue:
        for venue in sorted(missing_by_venue):
            lines.append(f"- `{venue}`: `{missing_by_venue[venue]}`")
    else:
        lines.append("- `unknown`: current Phase 5.1ak summary was not available")
    lines.extend(
        [
            "",
            "Accepted next inputs:",
            "- A real sanitized Phase 5.1al forward-refresh input captured with deterministic target keys and source truth at event time.",
            "- A validated redacted Phase 5.1ad mapping from `source_record_sha256` to `canonical_group_id` or `order_key`.",
            "- Materially new directly target-linkable Phase 5.1aj private/native rows.",
            "- Complete sanitized Phase 5.1ab Lighter event-time pressure rows.",
            "",
            "Rejected inputs:",
            "- Documentation-only limits, configured caps, account tiers, empty headers, or current-only snapshots.",
            "- Source links inferred from time, price, size, venue role, account role, or proximity.",
            "- Raw private identifiers, credentials, signed payloads, or authorization material.",
            "- Any live/canary/capital/risk authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _delta(current: Any, previous: Any) -> int | None:
    current_int = _int_or_none(current)
    previous_int = _int_or_none(previous)
    if current_int is None or previous_int is None:
        return None
    return current_int - previous_int


def _current_blocker_counts(summary: dict[str, Any]) -> dict[str, Any]:
    blocker = summary.get("current_blocker") or {}
    if not isinstance(blocker, dict):
        blocker = {}
    return {
        "lighter_pressure_missing": blocker.get("lighter_native_limit_capture_target_missing_count"),
        "lighter_pressure_ready": blocker.get("lighter_native_limit_capture_target_ready_count"),
        "native_role_missing": blocker.get("native_role_capture_target_missing_count"),
        "native_role_ready": blocker.get("native_role_capture_target_ready_count"),
        "unrecoverable_from_local_artifacts": blocker.get("unrecoverable_from_local_artifacts_count"),
    }


def _build_phase51am_delta(previous_record: dict[str, Any] | None, current_summary: dict[str, Any]) -> dict[str, Any] | None:
    if previous_record is None:
        return None
    previous_summary = previous_record["summary"]
    previous_counts = _current_blocker_counts(previous_summary)
    current_counts = _current_blocker_counts(current_summary)
    selected_route_changed = previous_summary.get("selected_route") != current_summary.get("selected_route")
    control_status_changed = previous_summary.get("control_status") != current_summary.get("control_status")
    ready_route_count_delta = _delta(current_summary.get("ready_route_count"), previous_summary.get("ready_route_count"))
    count_deltas = {
        f"{key}_delta": _delta(current_counts.get(key), previous_counts.get(key))
        for key in sorted(current_counts)
    }
    blocker_counts_changed = any(value not in (None, 0) for value in count_deltas.values())
    route_readiness_changed = ready_route_count_delta not in (None, 0) or selected_route_changed or control_status_changed
    if route_readiness_changed or blocker_counts_changed:
        staleness_status = "ROUTE_OR_BLOCKER_CHANGED"
        optimization_signal = "prioritize_changed_ready_route_or_blocker_delta"
    elif current_summary.get("selected_route") == "none":
        staleness_status = "UNCHANGED_NO_READY_ROUTE"
        optimization_signal = "avoid_duplicate_local_mining_and_keep_source_owner_handoff"
    else:
        staleness_status = "UNCHANGED_READY_ROUTE"
        optimization_signal = "execute_existing_ready_route_before_expanding_board"
    return {
        "previous_summary_path": previous_record["path"],
        "previous_summary_sha256": previous_record["sha256"],
        "previous_run_id": previous_summary.get("run_id"),
        "previous_generated_at_utc": previous_summary.get("generated_at_utc"),
        "previous_selected_route": previous_summary.get("selected_route"),
        "current_selected_route": current_summary.get("selected_route"),
        "selected_route_changed": selected_route_changed,
        "control_status_changed": control_status_changed,
        "ready_route_count_delta": ready_route_count_delta,
        "blocker_counts_changed": blocker_counts_changed,
        "route_readiness_changed": route_readiness_changed,
        "staleness_status": staleness_status,
        "optimization_signal": optimization_signal,
        "count_deltas": count_deltas,
    }


def _build_workflow_optimization_ledger(
    *,
    selected_route: dict[str, Any] | None,
    route_decisions: list[dict[str, Any]],
    work_packets: list[dict[str, Any]],
    current_blocker: dict[str, Any],
    phase51am_delta: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    ready_route_count = len(
        [decision for decision in route_decisions if str(decision.get("route_status", "")).startswith("READY")]
    )
    blocked_route_count = len([decision for decision in route_decisions if decision.get("route_status") == "BLOCKED"])
    selected_route_name = selected_route["route"] if selected_route is not None else "none"
    base_context = {
        "selected_route": selected_route_name,
        "ready_route_count": ready_route_count,
        "blocked_route_count": blocked_route_count,
        "subagent_work_packet_count": len(work_packets),
        "phase51ak_run_id": current_blocker.get("phase51ak_run_id"),
    }
    records = [
        {
            **base_context,
            "optimization_key": "continuous_reclassification",
            "status": "ACTIVE",
            "trigger": "after_every_blocker_task_or_new_source_owner_artifact",
            "action": "rerun_phase51am_and_refresh_route_decision_ledger",
            "rationale": "route readiness can change after any supplied mapping, direct row, pressure row, or forward-refresh pack",
            "audit_requirement": "compare new selected_route, ready_route_count, and blocker counts against the previous Phase 5.1am summary",
        },
        {
            **base_context,
            "optimization_key": "dynamic_board_sizing",
            "status": "ACTIVE",
            "trigger": "phase51am_route_selection",
            "action": (
                "collapse_to_systems_implementer_plus_independent_auditor"
                if selected_route is not None
                else "dispatch_source_owner_route_packets_plus_independent_auditor"
            ),
            "rationale": "use the smallest board that advances the selected route while keeping independent audit coverage",
            "audit_requirement": "close stale agents and avoid duplicate mandates before the next route execution",
        },
        {
            **base_context,
            "optimization_key": "anti_duplication_guard",
            "status": "ACTIVE",
            "trigger": "all_routes_blocked_without_material_new_input",
            "action": "block_repeated_retrospective_local_mining_and_request_source_owner_truth",
            "rationale": "Phase 5.1ak classifies the remaining blocker as unrecoverable from current local artifacts",
            "audit_requirement": "require a material-change reason before running another retrospective local/read-only source lane",
        },
        {
            **base_context,
            "optimization_key": "evidence_gate_priority",
            "status": "ACTIVE",
            "trigger": "multiple_ready_routes",
            "action": "prefer_forward_refresh_then_validated_mapping_then_direct_private_rows_then_lighter_pressure",
            "rationale": "route priority favors complete event-time target/source truth before narrower retained-target repair paths",
            "audit_requirement": "if priority order is overridden, record the source-owner artifact and safety reason",
        },
        {
            **base_context,
            "optimization_key": "post_task_audit_loop",
            "status": "ACTIVE",
            "trigger": "after_any_packet_completion_or_downstream_gate_run",
            "action": "run_independent_audit_then_rerun_phase51am",
            "rationale": "workflow optimization is continuous, not a one-time plan",
            "audit_requirement": "auditor answers whether the current move remains the highest-leverage safe continuation",
        },
    ]
    if selected_route is None:
        records.append(
            {
                **base_context,
                "optimization_key": "no_route_handoff_compression",
                "status": "ACTIVE",
                "trigger": "selected_route_none",
                "action": "use_source_owner_request_md_as_single_external_handoff",
                "rationale": "one consolidated source-owner request reduces duplicated prompts and avoids drifting requirements",
                "audit_requirement": "confirm the request still lists only admissible source-owner inputs",
            }
        )
    else:
        records.append(
            {
                **base_context,
                "optimization_key": "ready_route_fast_path",
                "status": "ACTIVE",
                "trigger": "selected_route_ready",
                "action": "execute_selected_route_packet_before_dispatching_lower_priority_packets",
                "rationale": "ready evidence should be validated before asking agents to pursue blocked lower-priority lanes",
                "audit_requirement": "validate the selected route through its named Phase 5.1 gate before any readiness claim",
            }
        )
    if phase51am_delta is not None:
        records.append(
            {
                **base_context,
                "optimization_key": "previous_run_delta_monitor",
                "status": "ACTIVE",
                "trigger": "phase51am_previous_summary_available",
                "action": phase51am_delta["optimization_signal"],
                "rationale": phase51am_delta["staleness_status"],
                "audit_requirement": "explain any unchanged no-route repeat before dispatching more local mining work",
                "previous_run_id": phase51am_delta.get("previous_run_id"),
                "selected_route_changed": phase51am_delta.get("selected_route_changed"),
                "blocker_counts_changed": phase51am_delta.get("blocker_counts_changed"),
            }
        )
    return records


def build_nonlive_executive_orchestrator(
    *,
    repo_root: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
    phase51ak_summary: Path | None,
    previous_phase51am_summary: Path | None,
    source_owner_intake_manifest: Path | None,
    phase51al_summaries: list[Path],
    validated_mappings: list[Path],
    phase51aj_source_specs: list[str],
    phase51ab_pressure_jsonls: list[Path],
) -> Path:
    repo_root = repo_root.resolve()
    _check_run_id(run_id)
    output_root = _resolve_path(output_root, repo_root)
    if output_root.exists() and _path_has_symlink(output_root):
        raise ValueError(f"output root must not traverse symlinks: {output_root}")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_ak = [phase51ak_summary] if phase51ak_summary is not None else []
    phase51ak_records = _collect_json_summaries(
        repo_root,
        explicit_ak,
        "runs/phase51ak_blocker_resolution_runner/*/phase51ak_blocker_resolution_summary.json",
        label="Phase 5.1ak summary",
    )
    current_blocker = _extract_current_blocker(phase51ak_records[0] if phase51ak_records else None)
    source_owner_intake = _load_source_owner_intake_manifest(repo_root, source_owner_intake_manifest)
    phase51al_summaries = _dedupe_paths(phase51al_summaries + source_owner_intake["phase51al_summaries"])
    validated_mappings = _dedupe_paths(validated_mappings + source_owner_intake["validated_mappings"])
    phase51aj_source_specs = _dedupe_strings(phase51aj_source_specs + source_owner_intake["phase51aj_source_json"])
    phase51ab_pressure_jsonls = _dedupe_paths(phase51ab_pressure_jsonls + source_owner_intake["phase51ab_pressure_jsonls"])

    phase51al_records = _collect_json_summaries(
        repo_root,
        phase51al_summaries,
        "runs/phase51al_forward_refresh_capture_gate/*/phase51al_forward_refresh_capture_summary.json",
        label="Phase 5.1al summary",
    )
    route_decisions = [
        _classify_forward_refresh(repo_root, phase51al_records),
        _classify_validated_mapping(repo_root, validated_mappings, current_blocker),
        _classify_direct_private_rows(repo_root, phase51aj_source_specs, current_blocker),
        _classify_lighter_pressure(repo_root, phase51ab_pressure_jsonls, current_blocker),
    ]
    selected_route = _select_route(route_decisions)
    work_packets = (
        _build_ready_route_packets(selected_route, current_blocker)
        if selected_route is not None
        else _build_no_route_work_packets(current_blocker)
    )
    previous_phase51am_record = _discover_previous_phase51am_summary(
        repo_root,
        out_dir,
        previous_phase51am_summary,
    )

    summary_path = out_dir / "phase51am_nonlive_executive_orchestrator_summary.json"
    decision_ledger_path = out_dir / "phase51am_route_decision_ledger.jsonl"
    work_packet_path = out_dir / "subagent_work_packets.jsonl"
    workflow_optimization_ledger_path = out_dir / "workflow_optimization_ledger.jsonl"
    source_owner_intake_status_path = out_dir / "source_owner_intake_status.json"
    source_owner_intake_template_path = out_dir / "source_owner_intake_manifest.template.json"
    source_owner_request_path = out_dir / "source_owner_request.md"
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    manifest_path = out_dir / "manifest.json"

    source_owner_text = _build_source_owner_request(current_blocker, selected_route)
    _write_jsonl(decision_ledger_path, route_decisions)
    _write_jsonl(work_packet_path, work_packets)
    subagent_prompt_pack_path, subagent_prompt_pack_index_path, subagent_prompt_paths = _write_subagent_prompt_pack(
        out_dir,
        work_packets,
    )
    _write_json(
        source_owner_intake_status_path,
        _source_owner_intake_status(
            source_owner_intake,
            phase51al_summaries,
            validated_mappings,
            phase51aj_source_specs,
            phase51ab_pressure_jsonls,
        ),
    )
    _write_json(source_owner_intake_template_path, _source_owner_intake_template())
    _write_text(source_owner_request_path, source_owner_text)

    ready_route_count = len([decision for decision in route_decisions if str(decision.get("route_status", "")).startswith("READY")])
    selected_route_name = selected_route["route"] if selected_route is not None else "none"
    control_status = "READY_TO_EXECUTE_SELECTED_ROUTE" if selected_route is not None else "AWAITING_SOURCE_OWNER_INPUT"
    gate_reason = (
        f"phase51am_{selected_route_name}_route_ready_nonlive_hold"
        if selected_route is not None
        else "phase51am_no_admissible_source_owner_input_work_packets_emitted_nonlive_hold"
    )
    next_required_action = (
        selected_route.get("next_required_action")
        if selected_route is not None
        else "dispatch_subagent_work_packets_and_obtain_source_owner_truth"
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
        "control_status": control_status,
        "selected_route": selected_route_name,
        "ready_route_count": ready_route_count,
        "route_count": len(route_decisions),
        "route_status_counts": _status_counts(route_decisions, "route_status"),
        "current_blocker": current_blocker,
        "source_owner_native_role_evidence_ready": current_blocker.get("source_owner_native_role_evidence_ready"),
        "phase51_source_owner_blocker_status": current_blocker.get("phase51_source_owner_blocker_status"),
        "lighter_pressure_unavailable_governance_accepted": current_blocker.get(
            "lighter_pressure_unavailable_governance_accepted"
        ),
        "h_i_feature_matrix_deferred": current_blocker.get("h_i_feature_matrix_deferred"),
        "decision_ledger_path": str(decision_ledger_path),
        "subagent_work_packet_path": str(work_packet_path),
        "subagent_work_packet_count": len(work_packets),
        "subagent_prompt_pack_path": str(subagent_prompt_pack_path),
        "subagent_prompt_pack_index_path": str(subagent_prompt_pack_index_path),
        "subagent_prompt_count": len(subagent_prompt_paths),
        "workflow_optimization_ledger_path": str(workflow_optimization_ledger_path),
        "workflow_optimization_status": "CONTINUOUS_ACTIVE",
        "source_owner_intake_status_path": str(source_owner_intake_status_path),
        "source_owner_intake_manifest_supplied": source_owner_intake.get("manifest_path") is not None,
        "source_owner_intake_manifest_template_path": str(source_owner_intake_template_path),
        "source_owner_request_path": str(source_owner_request_path),
        "autonomous_continuation_status": (
            "selected_route_work_packets_emitted"
            if selected_route is not None
            else "source_owner_and_subagent_work_packets_emitted"
        ),
        "implementation_route_blocked": selected_route is None,
        "next_required_action": next_required_action,
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
    if selected_route is not None:
        summary["selected_route_decision"] = selected_route
    phase51am_delta = _build_phase51am_delta(previous_phase51am_record, summary)
    if phase51am_delta is not None:
        summary["phase51am_delta"] = phase51am_delta
    workflow_optimization_records = _build_workflow_optimization_ledger(
        selected_route=selected_route,
        route_decisions=route_decisions,
        work_packets=work_packets,
        current_blocker=current_blocker,
        phase51am_delta=phase51am_delta,
    )
    summary["workflow_optimization_action_count"] = len(workflow_optimization_records)
    _write_jsonl(workflow_optimization_ledger_path, workflow_optimization_records)
    _write_json(summary_path, summary)

    artifact_paths = [
        decision_ledger_path,
        work_packet_path,
        subagent_prompt_pack_index_path,
        *subagent_prompt_paths,
        workflow_optimization_ledger_path,
        source_owner_intake_status_path,
        source_owner_intake_template_path,
        source_owner_request_path,
        summary_path,
    ]
    artifact_index = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "artifacts": _artifact_infos(out_dir, artifact_paths),
        "no_live_flag": True,
        "approved_for_live": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }
    _write_json(artifact_index_path, artifact_index)
    artifact_paths.append(artifact_index_path)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "control_status": control_status,
        "selected_route": selected_route_name,
        "summary_path": str(summary_path),
        "artifacts": _artifact_infos(out_dir, artifact_paths),
        "clears_phase51_blockers": False,
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
    }
    _write_json(manifest_path, manifest)
    return out_dir


def _status_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--phase51ak-summary", type=Path)
    parser.add_argument("--previous-phase51am-summary", type=Path)
    parser.add_argument("--source-owner-intake-manifest", type=Path)
    parser.add_argument("--phase51al-summary", action="append", type=Path, default=[])
    parser.add_argument("--validated-mapping", action="append", type=Path, default=[])
    parser.add_argument("--phase51aj-source-json", action="append", default=[], help="venue=/path/to/file")
    parser.add_argument("--phase51ab-pressure-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AM-NONLIVE-EXECUTIVE-ORCHESTRATOR-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int)
    args = parser.parse_args()

    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_nonlive_executive_orchestrator(
            repo_root=args.repo_root,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
            phase51ak_summary=args.phase51ak_summary,
            previous_phase51am_summary=args.previous_phase51am_summary,
            source_owner_intake_manifest=args.source_owner_intake_manifest,
            phase51al_summaries=args.phase51al_summary,
            validated_mappings=args.validated_mapping,
            phase51aj_source_specs=args.phase51aj_source_json,
            phase51ab_pressure_jsonls=args.phase51ab_pressure_jsonl,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail closed with a concise error
        print(f"phase51am_nonlive_executive_orchestrator: ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"phase51am_nonlive_executive_orchestrator: wrote {out_dir}")
    print("phase51am_nonlive_executive_orchestrator: status HOLD (non-live executive control only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
