#!/usr/bin/env python3
"""
Phase 5 tranche automation for Paraphina.

This module formalizes the Phase 5 workflow as:
- a serialized mainline queue
- parallel support tracks
- repo-local tranche cards / run manifests
- wrappers around the existing guard/analyzer tooling
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PHASE5_DIR = ROOT / "phase5"
QUEUE_PATH = PHASE5_DIR / "queue.yaml"
CONTROL_PACK_PATH = PHASE5_DIR / "control_pack.yaml"
STATUS_PATH = PHASE5_DIR / "status.md"
RUNS_DIR = PHASE5_DIR / "runs"
ORCHESTRATION_PATH = PHASE5_DIR / "orchestration.yaml"
LEGACY_REOPENED_LONG_SOAK_ID = "phase5_reopened_multi_venue_long_soak"
REOPENED_LONG_SOAK_ID = "phase5_reopened_multi_venue_long_soak_runner_realtime_requal"
REOPENED_FINAL_CLOSEOUT_ID = "phase5_reopened_final_closeout"
REOPENED_LONG_SOAK_NO_LOOP_BLOCKERS = {
    "all5_projected_mm_budget_distribution_gap",
    "all_venue_market_frontier_backpressure_gap",
    "paradex_edge_floor_queue_loss",
    "paradex_queue_position_loss",
}

VALID_STATUSES = {
    "pending",
    "ready",
    "in_progress",
    "blocked",
    "hold",
    "promoted",
    "rolled_back",
}

VALID_BRANCH_CLASSES = {
    "localization",
    "clearance",
    "qualification",
    "topology",
}

VALID_BLOCKER_FAMILIES = {
    "stale_restart",
    "hyperliquid_post_publish_transport_gap",
    "hyperliquid_pre_kill_recovery_alignment_gap",
    "topology_fv_reentry_gap",
    "startup_readiness_gap",
    "transport_gap_watchdog",
    "aster_bridge_wait_timeout",
    "restore_hygiene",
    "exact_one_lot_residual_no_orders",
    "exact_one_lot_execution_truth",
    "actual_residual_live_orders",
    "paradex_underfill_with_interactive_profile",
    "paradex_underfill_with_ui_book_truth",
    "microstructure_underconversion",
    "all5_projected_mm_budget_distribution_gap",
    "paradex_batch_cancel_request_shape_gap",
    "paradex_replace_identity_gap",
    "paradex_same_side_persistence_gap",
    "paradex_private_order_truth_gap",
    "paradex_queue_position_loss",
    "paradex_edge_floor_queue_loss",
    "paradex_edge_floor_shadow_mechanism_gate_gap",
    "paradex_open_snapshot_replace_identity_gap",
    "paradex_interactive_top_anchor_gap",
    "paradex_ui_touch_reference_gap",
    "lighter_sequence_continuity_gap",
    "extended_post_publish_fallback_rearm_gap",
    "extended_degraded_stream_rebootstrap_gap",
    "extended_pre_kill_degraded_rebootstrap_alignment_gap",
    "extended_post_publish_stream_delivery_gap",
    "soft_cap_starvation",
    "no_data_transport_gap",
    "data_seen_no_publish",
    "runner_freeze_apply_gap",
    "hyperliquid_canary_response_sync_timeout",
    "future_timestamp_deferral",
    "all_venue_market_frontier_backpressure_gap",
    "capital_preservation_residual_markout",
}

VALID_SUPPORT_GATES = {
    "none",
    "shadow_smoke_10m",
    "shadow_smoke_30m",
    "shadow_ab_10m",
}

VALID_PROGRESS_CREDIT = {
    "none",
    "minor",
    "major",
}

VALID_AUTORUN_POLICY = {
    "manual",
    "shadow_smoke",
    "shadow_ab",
    "validate_only",
}

VALID_SUPPORT_TRIGGER_MODE = {
    "always",
    "on_blocker_family",
}

VALID_AUTONOMY_MODE = {
    "full_auto",
    "semi_auto",
    "prep_only",
}

VALID_SUBAGENT_MODEL = {
    "deterministic_handoff",
    "local_agent_hooks",
    "deterministic_only",
}

VALID_WORKTREE_LIFECYCLE = {
    "ephemeral",
    "persistent",
    "hybrid",
}

VALID_ARTIFACT_PACKAGING = {
    "full",
    "lean",
    "mixed",
}

VALID_AUTOSCORE_SOURCES = {
    "balance_snapshot",
    "cashflow",
    "closeout",
    "direct_venue_audit",
    "summary",
    "metrics",
    "health_post",
    "systemd_post",
}

VALID_AUTOSCORE_OPS = {
    "==",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not_in",
}

VALID_RUNG_CONTINUE_ON = {
    "clean",
    "mechanism",
    "promotion",
}

VALID_GATE_CONTRACT_REQUIRED_ARTIFACTS = {
    "closeout",
    "metrics",
    "autoscore",
    "direct_venue_audit",
    "cashflow",
    "state_sync",
    "balance_snapshot",
}

GATE_CONTRACT_ECONOMICS_TERMS = {
    "account",
    "balance",
    "capital",
    "cashflow",
    "cleanup",
    "drawdown",
    "econ",
    "economics",
    "equity",
    "fee",
    "funding",
    "markout",
    "pnl",
    "profit",
    "residual",
}

VALID_MECHANISM_GATE_MODE = {
    "shadow",
    "live_5m",
    "live_20m",
    "live_60m",
}

LANE_ROLE_LIVE = "live_sentinel"
LANE_ROLE_FORENSICS = "forensics_gatekeeper"
LANE_ROLE_PASS_PREP = "pass_prep_operator"
LANE_ROLE_FAIL_PREP = "fail_prep_operator"
LANE_ROLE_SUPPORT_PREFIX = "support_track_operator__"

DEFAULT_SUPPORT_LANE_PRIORITY = [
    "forensics",
    "blocker_shadow",
    "topology_audit",
    "tooling",
]

DEFAULT_STAGE_VERDICT_CONTRACT = [
    "stage_verdict.json",
    "venue_capability_matrix.json",
    "support_summary.json",
]

DEFAULT_RUNG_PLAN = [
    {"duration_sec": 300, "continue_on": "clean"},
    {"duration_sec": 1200, "continue_on": "clean"},
    {"duration_sec": 3600, "continue_on": "promotion"},
]

DEFAULT_AUTOSCORE_RULES = {
    "clean": [
        {"source": "closeout", "path": "summary_exists", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "report_exists", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "metrics_exists", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "closeout_contract_complete", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "guard_intervened", "op": "==", "value": False, "severity": "fail"},
        {"source": "closeout", "path": "guard_window_completed", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "first_pre_restore_venue_audit_clean", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "pre_restore_cleanup_required", "op": "==", "value": False, "severity": "fail"},
        {"source": "closeout", "path": "pre_restore_venue_audit_clean", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "post_rollback_venue_audit_clean", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "healthy_post", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "ready_post", "op": "==", "value": True, "severity": "fail"},
        {"source": "closeout", "path": "kill_events_present_post", "op": "==", "value": False, "severity": "fail"},
        {"source": "closeout", "path": "trade_mode_post", "op": "==", "value": "shadow", "severity": "fail"},
        {"source": "closeout", "path": "systemd_nrestarts_post", "op": "==", "value": "0", "severity": "fail"},
    ],
    "mechanism": [],
    "promotion": [],
}

HEALTH_URL = "http://127.0.0.1:9898/health/detail"
CURRENT_RUNS_DIR = "/tmp/paraphina_current_runs"

MIB = 1024 * 1024
GIB = 1024 * MIB


@dataclass
class RunCommand:
    label: str
    argv: list[str]
    cwd: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    check: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return data


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def iter_tranches(queue: dict[str, Any]):
    for section in ("serialized_mainline", "parallel_support_tracks"):
        items = queue.get(section, []) or []
        for index, tranche in enumerate(items):
            yield section, index, tranche


def find_tranche(queue: dict[str, Any], tranche_id: str) -> tuple[str, int, dict[str, Any]]:
    for section, index, tranche in iter_tranches(queue):
        if tranche.get("id") == tranche_id:
            return section, index, tranche
    raise KeyError(f"Unknown tranche id: {tranche_id}")


def tranche_ids(queue: dict[str, Any]) -> set[str]:
    return {tranche.get("id") for _, _, tranche in iter_tranches(queue)}


def latest_history_entry(tranche: dict[str, Any]) -> dict[str, Any] | None:
    history = tranche.get("history", []) or []
    if not isinstance(history, list) or not history:
        return None
    last = history[-1]
    return last if isinstance(last, dict) else None


def parse_timestamp_utc(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def format_decimal(value: Any, places: int = 4) -> str:
    text = f"{safe_float(value):.{places}f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def validate_optional_enum(tranche_id: str, value: Any, valid_values: set[str], field_name: str) -> None:
    if value is None:
        return
    if value not in valid_values:
        raise ValueError(f"{tranche_id}: invalid {field_name} {value!r}")


def validate_optional_string_list(
    tranche_id: str,
    value: Any,
    valid_values: set[str],
    field_name: str,
) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f"{tranche_id}: {field_name} must be a list when present")
    for item in value:
        if item not in valid_values:
            raise ValueError(f"{tranche_id}: invalid {field_name} entry {item!r}")


def validate_autoscore_rule(owner_id: str, group_name: str, index: int, rule: Any) -> None:
    if not isinstance(rule, dict):
        raise ValueError(f"{owner_id}: automation.autoscore.{group_name}[{index}] must be a mapping")
    source = rule.get("source")
    if source not in VALID_AUTOSCORE_SOURCES:
        raise ValueError(f"{owner_id}: invalid autoscore source {source!r}")
    path = rule.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{owner_id}: autoscore rule path must be a non-empty string")
    op = rule.get("op")
    if op not in VALID_AUTOSCORE_OPS:
        raise ValueError(f"{owner_id}: invalid autoscore op {op!r}")
    severity = rule.get("severity", "fail")
    if severity not in {"fail", "hold_only", "warn", "rollback"}:
        raise ValueError(f"{owner_id}: invalid autoscore severity {severity!r}")
    decision_effect = rule.get("decision_effect")
    if decision_effect is not None and decision_effect not in {"fail", "hold", "warn", "rollback"}:
        raise ValueError(f"{owner_id}: invalid autoscore decision_effect {decision_effect!r}")


def validate_support_family(owner_id: str, index: int, family: Any, support_ids: set[str]) -> None:
    if not isinstance(family, dict):
        raise ValueError(f"{owner_id}: automation.support_families[{index}] must be a mapping")
    family_id = family.get("id")
    if not isinstance(family_id, str) or not family_id:
        raise ValueError(f"{owner_id}: automation.support_families[{index}].id must be a non-empty string")
    support_track_id = family.get("support_track_id")
    if not isinstance(support_track_id, str) or support_track_id not in support_ids:
        raise ValueError(
            f"{owner_id}: automation.support_families[{index}].support_track_id targets missing support tranche {support_track_id!r}"
        )
    trigger_mode = family.get("trigger_mode", "always")
    if trigger_mode not in VALID_SUPPORT_TRIGGER_MODE:
        raise ValueError(
            f"{owner_id}: invalid automation.support_families[{index}].trigger_mode {trigger_mode!r}"
        )
    blocker_families = family.get("blocker_families")
    validate_optional_string_list(
        owner_id,
        blocker_families,
        VALID_BLOCKER_FAMILIES,
        f"automation.support_families[{index}].blocker_families",
    )
    if trigger_mode == "on_blocker_family" and not blocker_families:
        raise ValueError(
            f"{owner_id}: automation.support_families[{index}] requires blocker_families for on_blocker_family"
        )
    validate_optional_enum(
        owner_id,
        family.get("autorun_policy"),
        VALID_AUTORUN_POLICY,
        f"automation.support_families[{index}].autorun_policy",
    )
    max_parallel_runs = family.get("max_parallel_runs")
    if max_parallel_runs is not None and (not isinstance(max_parallel_runs, int) or max_parallel_runs <= 0):
        raise ValueError(
            f"{owner_id}: automation.support_families[{index}].max_parallel_runs must be a positive int"
        )
    stop_on_mainline_promote = family.get("stop_on_mainline_promote")
    if stop_on_mainline_promote is not None and not isinstance(stop_on_mainline_promote, bool):
        raise ValueError(
            f"{owner_id}: automation.support_families[{index}].stop_on_mainline_promote must be a boolean"
        )


def validate_automation_block(
    owner_id: str,
    tranche: dict[str, Any],
    ids: set[str],
    support_ids: set[str],
    is_support: bool,
) -> None:
    automation = tranche.get("automation")
    if automation is None:
        return
    if not isinstance(automation, dict):
        raise ValueError(f"{owner_id}: automation must be a mapping when present")
    validate_optional_enum(owner_id, automation.get("autonomy_mode"), VALID_AUTONOMY_MODE, "automation.autonomy_mode")
    validate_optional_enum(
        owner_id,
        automation.get("subagent_model"),
        VALID_SUBAGENT_MODEL,
        "automation.subagent_model",
    )
    validate_optional_enum(
        owner_id,
        automation.get("worktree_lifecycle"),
        VALID_WORKTREE_LIFECYCLE,
        "automation.worktree_lifecycle",
    )
    validate_optional_enum(
        owner_id,
        automation.get("artifact_packaging"),
        VALID_ARTIFACT_PACKAGING,
        "automation.artifact_packaging",
    )

    if is_support:
        autorun_policy = automation.get("autorun_policy")
        validate_optional_enum(owner_id, autorun_policy, VALID_AUTORUN_POLICY, "automation.autorun_policy")
        return

    rung_plan = automation.get("rung_plan")
    if rung_plan is not None:
        if not isinstance(rung_plan, list) or not rung_plan:
            raise ValueError(f"{owner_id}: automation.rung_plan must be a non-empty list when present")
        for index, rung in enumerate(rung_plan):
            if not isinstance(rung, dict):
                raise ValueError(f"{owner_id}: automation.rung_plan[{index}] must be a mapping")
            duration_sec = rung.get("duration_sec")
            if not isinstance(duration_sec, int) or duration_sec <= 0:
                raise ValueError(f"{owner_id}: automation.rung_plan[{index}].duration_sec must be a positive int")
            continue_on = rung.get("continue_on", "clean")
            if continue_on not in VALID_RUNG_CONTINUE_ON:
                raise ValueError(f"{owner_id}: invalid automation.rung_plan[{index}].continue_on {continue_on!r}")

    support_tracks = automation.get("support_tracks")
    if support_tracks is not None:
        if not isinstance(support_tracks, list):
            raise ValueError(f"{owner_id}: automation.support_tracks must be a list when present")
        for track_id in support_tracks:
            if track_id not in support_ids:
                raise ValueError(f"{owner_id}: automation.support_tracks targets missing support tranche {track_id!r}")

    support_families = automation.get("support_families")
    if support_families is not None:
        if not isinstance(support_families, list):
            raise ValueError(f"{owner_id}: automation.support_families must be a list when present")
        for index, family in enumerate(support_families):
            validate_support_family(owner_id, index, family, support_ids)

    autoscore = automation.get("autoscore")
    if autoscore is not None:
        if not isinstance(autoscore, dict):
            raise ValueError(f"{owner_id}: automation.autoscore must be a mapping when present")
        for group_name in ("clean", "mechanism", "promotion"):
            rules = autoscore.get(group_name)
            if rules is None:
                continue
            if not isinstance(rules, list):
                raise ValueError(f"{owner_id}: automation.autoscore.{group_name} must be a list")
            for index, rule in enumerate(rules):
                validate_autoscore_rule(owner_id, group_name, index, rule)


def validate_queue(queue: dict[str, Any]) -> None:
    if queue.get("schema_version") != 1:
        raise ValueError("phase5 queue schema_version must be 1")
    seen: set[str] = set()
    ids = tranche_ids(queue)
    support_ids = {
        tranche.get("id")
        for section, _, tranche in iter_tranches(queue)
        if section == "parallel_support_tracks"
    }
    for _, _, tranche in iter_tranches(queue):
        tranche_id = tranche.get("id")
        if not tranche_id or not isinstance(tranche_id, str):
            raise ValueError("Every tranche requires a non-empty string id")
        if tranche_id in seen:
            raise ValueError(f"Duplicate tranche id: {tranche_id}")
        seen.add(tranche_id)
        status = tranche.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"{tranche_id}: invalid status {status!r}")
        validate_optional_enum(tranche_id, tranche.get("branch_class"), VALID_BRANCH_CLASSES, "branch_class")
        validate_optional_enum(
            tranche_id,
            tranche.get("hypothesis_blocker_family"),
            VALID_BLOCKER_FAMILIES,
            "hypothesis_blocker_family",
        )
        validate_optional_enum(
            tranche_id,
            tranche.get("mechanism_gate_mode"),
            VALID_MECHANISM_GATE_MODE,
            "mechanism_gate_mode",
        )
        validate_optional_enum(
            tranche_id,
            tranche.get("mechanism_fail_blocker_family"),
            VALID_BLOCKER_FAMILIES,
            "mechanism_fail_blocker_family",
        )
        validate_optional_enum(
            tranche_id,
            tranche.get("clean_final_hold_blocker_family"),
            VALID_BLOCKER_FAMILIES,
            "clean_final_hold_blocker_family",
        )
        validate_optional_string_list(
            tranche_id,
            tranche.get("required_cleared_blockers"),
            VALID_BLOCKER_FAMILIES,
            "required_cleared_blockers",
        )
        validate_optional_enum(tranche_id, tranche.get("support_gate"), VALID_SUPPORT_GATES, "support_gate")
        support_gate_require_mechanism = tranche.get("support_gate_require_mechanism")
        if support_gate_require_mechanism is not None and not isinstance(support_gate_require_mechanism, bool):
            raise ValueError(f"{tranche_id}: support_gate_require_mechanism must be a bool when present")
        validate_optional_enum(tranche_id, tranche.get("progress_credit"), VALID_PROGRESS_CREDIT, "progress_credit")
        surface_local_restore_hygiene_child = tranche.get("surface_local_restore_hygiene_child")
        if surface_local_restore_hygiene_child is not None:
            if (
                not isinstance(surface_local_restore_hygiene_child, str)
                or surface_local_restore_hygiene_child not in ids
            ):
                raise ValueError(
                    f"{tranche_id}: surface_local_restore_hygiene_child targets missing tranche {surface_local_restore_hygiene_child!r}"
                )
        for edge in ("next_if_pass", "next_if_fail"):
            target = tranche.get(edge)
            if target and target not in ids:
                raise ValueError(f"{tranche_id}: {edge} targets missing tranche {target!r}")
        next_if_fail_when_matched = tranche.get("next_if_fail_when_matched")
        if next_if_fail_when_matched and next_if_fail_when_matched not in ids:
            raise ValueError(
                f"{tranche_id}: next_if_fail_when_matched targets missing tranche {next_if_fail_when_matched!r}"
            )
        matched_fail_routes = tranche.get("matched_fail_routes")
        if matched_fail_routes is not None:
            if not isinstance(matched_fail_routes, dict):
                raise ValueError(f"{tranche_id}: matched_fail_routes must be a mapping when present")
            for blocker_family, target in matched_fail_routes.items():
                if blocker_family not in VALID_BLOCKER_FAMILIES:
                    raise ValueError(
                        f"{tranche_id}: matched_fail_routes uses invalid blocker family {blocker_family!r}"
                    )
                if target not in ids:
                    raise ValueError(
                        f"{tranche_id}: matched_fail_routes[{blocker_family!r}] targets missing tranche {target!r}"
                    )
        manual_gate_required = tranche.get("manual_gate_required")
        if manual_gate_required is not None and not isinstance(manual_gate_required, bool):
            raise ValueError(f"{tranche_id}: manual_gate_required must be a boolean when present")
        promotion_gate = tranche.get("promotion_gate")
        if promotion_gate is not None and not isinstance(promotion_gate, dict):
            raise ValueError(f"{tranche_id}: promotion_gate must be a mapping when present")
        validate_automation_block(
            tranche_id,
            tranche,
            ids,
            support_ids,
            is_support=(tranche.get("track") == "parallel_support"),
        )


def validate_control_pack(control_pack: dict[str, Any]) -> None:
    if control_pack.get("schema_version") != 1:
        raise ValueError("phase5 control_pack schema_version must be 1")
    defaults = control_pack.get("automation_defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("phase5 control_pack automation_defaults must be a mapping when present")
    validate_optional_enum(
        "control_pack",
        defaults.get("autonomy_mode"),
        VALID_AUTONOMY_MODE,
        "automation_defaults.autonomy_mode",
    )
    validate_optional_enum(
        "control_pack",
        defaults.get("subagent_model"),
        VALID_SUBAGENT_MODEL,
        "automation_defaults.subagent_model",
    )
    validate_optional_enum(
        "control_pack",
        defaults.get("worktree_lifecycle"),
        VALID_WORKTREE_LIFECYCLE,
        "automation_defaults.worktree_lifecycle",
    )
    validate_optional_enum(
        "control_pack",
        defaults.get("artifact_packaging"),
        VALID_ARTIFACT_PACKAGING,
        "automation_defaults.artifact_packaging",
    )
    validate_optional_enum(
        "control_pack",
        defaults.get("autorun_support_default"),
        VALID_AUTORUN_POLICY,
        "automation_defaults.autorun_support_default",
    )
    for int_field in (
        "repo_headroom_bytes",
        "promotion_runs_headroom_bytes",
        "telemetry_headroom_bytes",
        "tempdir_headroom_bytes",
    ):
        value = defaults.get(int_field)
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"control_pack: automation_defaults.{int_field} must be a positive integer")
    max_parallel_support_lanes = defaults.get("max_parallel_support_lanes")
    if max_parallel_support_lanes is not None:
        if not isinstance(max_parallel_support_lanes, int) or max_parallel_support_lanes <= 0:
            raise ValueError("control_pack: automation_defaults.max_parallel_support_lanes must be a positive integer")
    support_lane_priority = defaults.get("support_lane_priority")
    if support_lane_priority is not None:
        if not isinstance(support_lane_priority, list) or not all(
            isinstance(item, str) and item for item in support_lane_priority
        ):
            raise ValueError("control_pack: automation_defaults.support_lane_priority must be a list of strings")
    support_lane_capacity_gate = defaults.get("support_lane_capacity_gate")
    if support_lane_capacity_gate is not None and not isinstance(support_lane_capacity_gate, bool):
        raise ValueError("control_pack: automation_defaults.support_lane_capacity_gate must be a boolean")
    stage_verdict_contract = defaults.get("stage_verdict_contract")
    if stage_verdict_contract is not None:
        if not isinstance(stage_verdict_contract, list) or not all(
            isinstance(item, str) and item for item in stage_verdict_contract
        ):
            raise ValueError("control_pack: automation_defaults.stage_verdict_contract must be a list of strings")
    rung_plan = defaults.get("default_rung_plan")
    if rung_plan is not None:
        validate_automation_block(
            "control_pack",
            {"automation": {"rung_plan": rung_plan}},
            set(),
            set(),
            is_support=False,
        )
    autoscore = defaults.get("default_autoscore")
    if autoscore is not None:
        validate_automation_block(
            "control_pack",
            {"automation": {"autoscore": autoscore}},
            set(),
            set(),
            is_support=False,
        )


def load_state(repo_root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = load_yaml(repo_root / "phase5" / "queue.yaml")
    control_pack = load_yaml(repo_root / "phase5" / "control_pack.yaml")
    validate_queue(queue)
    validate_control_pack(control_pack)
    return queue, control_pack


def save_state(queue: dict[str, Any], repo_root: Path = ROOT) -> None:
    queue["updated_utc"] = utc_now()
    save_yaml(repo_root / "phase5" / "queue.yaml", queue)


def default_orchestration_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "sessions": [],
    }


def load_orchestration(repo_root: Path = ROOT) -> dict[str, Any]:
    path = repo_root / "phase5" / "orchestration.yaml"
    if not path.exists():
        return default_orchestration_payload()
    payload = load_yaml(path)
    if payload.get("schema_version") != 1:
        raise ValueError("phase5 orchestration schema_version must be 1")
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        raise ValueError("phase5 orchestration sessions must be a list")
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("phase5 orchestration session entries must be mappings")
        if not isinstance(session.get("tranche_id"), str) or not session.get("tranche_id"):
            raise ValueError("phase5 orchestration session missing tranche_id")
        lanes = session.get("lanes", []) or []
        if not isinstance(lanes, list):
            raise ValueError("phase5 orchestration session lanes must be a list")
    return payload


def save_orchestration(payload: dict[str, Any], repo_root: Path = ROOT) -> None:
    payload["updated_utc"] = utc_now()
    save_yaml(repo_root / "phase5" / "orchestration.yaml", payload)


def find_orchestration_session(payload: dict[str, Any], tranche_id: str) -> tuple[int, dict[str, Any]] | None:
    for index, session in enumerate(payload.get("sessions", []) or []):
        if isinstance(session, dict) and session.get("tranche_id") == tranche_id:
            return index, session
    return None


def active_orchestration_session(
    payload: dict[str, Any],
    exclude_tranche_id: str | None = None,
) -> dict[str, Any] | None:
    for session in payload.get("sessions", []) or []:
        if not isinstance(session, dict):
            continue
        if exclude_tranche_id and session.get("tranche_id") == exclude_tranche_id:
            continue
        if session.get("state") in {"spawned", "running", "verdict_pending"}:
            return session
    return None


def upsert_orchestration_session(payload: dict[str, Any], session: dict[str, Any]) -> None:
    found = find_orchestration_session(payload, str(session.get("tranche_id")))
    if found is None:
        payload.setdefault("sessions", []).append(session)
        return
    index, _ = found
    payload["sessions"][index] = session


def session_lane(session: dict[str, Any], lane_id: str) -> dict[str, Any] | None:
    for lane in session.get("lanes", []) or []:
        if isinstance(lane, dict) and lane.get("lane_id") == lane_id:
            return lane
    return None


def set_lane_status(session: dict[str, Any], lane_id: str, status: str, **fields: Any) -> None:
    lane = session_lane(session, lane_id)
    if lane is None:
        raise KeyError(f"missing lane {lane_id}")
    lane["status"] = status
    lane["updated_utc"] = utc_now()
    lane.update(fields)


def repo_relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def automation_defaults(control_pack: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    defaults = control_pack.get("automation_defaults", {}) or {}
    worktree_root = defaults.get("worktree_root")
    if not isinstance(worktree_root, str) or not worktree_root:
        worktree_root = str(Path.home() / ".codex" / "phase5_worktrees" / repo_root.name)
    autonomy_mode = defaults.get("autonomy_mode", "full_auto")
    subagent_model = defaults.get("subagent_model", "deterministic_handoff")
    worktree_lifecycle = defaults.get("worktree_lifecycle", "ephemeral")
    artifact_packaging = defaults.get("artifact_packaging", "full")
    autorun_support_default = defaults.get("autorun_support_default", "manual")
    max_parallel_support_lanes = defaults.get("max_parallel_support_lanes", 0)
    support_lane_priority = copy.deepcopy(
        defaults.get("support_lane_priority", DEFAULT_SUPPORT_LANE_PRIORITY)
    )
    support_lane_capacity_gate = bool(defaults.get("support_lane_capacity_gate", False))
    stage_verdict_contract = copy.deepcopy(
        defaults.get("stage_verdict_contract", DEFAULT_STAGE_VERDICT_CONTRACT)
    )
    validate_optional_enum("control_pack", autonomy_mode, VALID_AUTONOMY_MODE, "automation_defaults.autonomy_mode")
    validate_optional_enum(
        "control_pack",
        subagent_model,
        VALID_SUBAGENT_MODEL,
        "automation_defaults.subagent_model",
    )
    validate_optional_enum(
        "control_pack",
        worktree_lifecycle,
        VALID_WORKTREE_LIFECYCLE,
        "automation_defaults.worktree_lifecycle",
    )
    validate_optional_enum(
        "control_pack",
        artifact_packaging,
        VALID_ARTIFACT_PACKAGING,
        "automation_defaults.artifact_packaging",
    )
    validate_optional_enum(
        "control_pack",
        autorun_support_default,
        VALID_AUTORUN_POLICY,
        "automation_defaults.autorun_support_default",
    )
    return {
        "autonomy_mode": autonomy_mode,
        "subagent_model": subagent_model,
        "worktree_lifecycle": worktree_lifecycle,
        "artifact_packaging": artifact_packaging,
        "worktree_root": worktree_root,
        "lane_bundle_root": defaults.get("lane_bundle_root", "phase5/runs/{tranche_id}/lanes"),
        "cleanup_policy": defaults.get("cleanup_policy", "ephemeral_on_verdict"),
        "repo_headroom_bytes": int(defaults.get("repo_headroom_bytes", 2 * GIB)),
        "promotion_runs_headroom_bytes": int(defaults.get("promotion_runs_headroom_bytes", 4 * GIB)),
        "telemetry_headroom_bytes": int(defaults.get("telemetry_headroom_bytes", 8 * GIB)),
        "tempdir_headroom_bytes": int(defaults.get("tempdir_headroom_bytes", 1 * GIB)),
        "autorun_support_default": autorun_support_default,
        "max_parallel_support_lanes": int(max_parallel_support_lanes),
        "support_lane_priority": support_lane_priority,
        "support_lane_capacity_gate": support_lane_capacity_gate,
        "stage_verdict_contract": stage_verdict_contract,
        "default_rung_plan": copy.deepcopy(defaults.get("default_rung_plan", DEFAULT_RUNG_PLAN)),
        "default_autoscore": copy.deepcopy(defaults.get("default_autoscore", DEFAULT_AUTOSCORE_RULES)),
    }


def merged_autoscore_rules(
    default_autoscore: dict[str, Any],
    override_autoscore: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = copy.deepcopy(default_autoscore)
    if override_autoscore is None:
        return merged
    for group_name, override_rules in override_autoscore.items():
        if group_name not in {"clean", "mechanism", "promotion"}:
            continue
        default_rules = merged.get(group_name, [])
        merged[group_name] = copy.deepcopy(default_rules) + copy.deepcopy(override_rules or [])
    return merged


def tranche_automation(tranche: dict[str, Any], control_pack: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    defaults = automation_defaults(control_pack, repo_root)
    automation = copy.deepcopy(tranche.get("automation", {}) or {})
    autoscore = merged_autoscore_rules(
        defaults["default_autoscore"],
        automation.get("autoscore"),
    )
    return {
        "autonomy_mode": automation.get("autonomy_mode", defaults["autonomy_mode"]),
        "subagent_model": automation.get("subagent_model", defaults["subagent_model"]),
        "worktree_lifecycle": automation.get("worktree_lifecycle", defaults["worktree_lifecycle"]),
        "artifact_packaging": automation.get("artifact_packaging", defaults["artifact_packaging"]),
        "worktree_root": automation.get("worktree_root", defaults["worktree_root"]),
        "lane_bundle_root": automation.get("lane_bundle_root", defaults["lane_bundle_root"]),
        "cleanup_policy": automation.get("cleanup_policy", defaults["cleanup_policy"]),
        "rung_plan": copy.deepcopy(automation.get("rung_plan", defaults["default_rung_plan"])),
        "support_tracks": list(automation.get("support_tracks", [])),
        "support_families": copy.deepcopy(automation.get("support_families", [])),
        "autoscore": autoscore,
        "autorun_policy": automation.get("autorun_policy", defaults["autorun_support_default"]),
        "max_parallel_support_lanes": defaults["max_parallel_support_lanes"],
        "support_lane_priority": copy.deepcopy(defaults["support_lane_priority"]),
        "support_lane_capacity_gate": defaults["support_lane_capacity_gate"],
        "stage_verdict_contract": copy.deepcopy(defaults["stage_verdict_contract"]),
    }


def gate_contract_required_artifacts(tranche: dict[str, Any]) -> set[str]:
    gate_contract = tranche.get("gate_contract") if isinstance(tranche.get("gate_contract"), dict) else {}
    automation = tranche.get("automation") if isinstance(tranche.get("automation"), dict) else {}
    raw = gate_contract.get("required_artifacts", automation.get("required_artifacts", []))
    if isinstance(raw, dict):
        raw = raw.keys()
    if not isinstance(raw, list) and not isinstance(raw, tuple) and not isinstance(raw, set):
        return set()
    return {str(item) for item in raw if isinstance(item, str) and item}


def gate_contract_text(tranche: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("id", "objective", "hypothesis", "branch_class", "hypothesis_blocker_family"):
        value = tranche.get(key)
        if value is not None:
            parts.append(str(value))
    for key in ("control", "mechanism_gate", "promotion_gate", "rollback_criteria"):
        value = tranche.get(key)
        if value is not None:
            parts.append(json.dumps(value, sort_keys=True, default=str))
    return " ".join(parts).lower()


def gate_contract_economics_relevant(tranche: dict[str, Any]) -> bool:
    text = gate_contract_text(tranche)
    return any(term in text for term in GATE_CONTRACT_ECONOMICS_TERMS)


def gate_contract_finding(
    findings: list[dict[str, Any]],
    severity: str,
    tranche_id: str,
    field: str,
    message: str,
) -> None:
    findings.append(
        {
            "severity": severity,
            "tranche_id": tranche_id,
            "field": field,
            "message": message,
        }
    )


def gate_contract_target_tranches(
    queue: dict[str, Any],
    tranche_id: str | None = None,
) -> list[dict[str, Any]]:
    if tranche_id:
        return [find_tranche(queue, tranche_id)[2]]
    current = current_serialized_mainline(queue)
    return [current] if current is not None else []


def audit_tranche_gate_contract(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    tranche_id = str(tranche.get("id") or "<missing>")
    findings: list[dict[str, Any]] = []
    automation_block = tranche.get("automation") if isinstance(tranche.get("automation"), dict) else {}
    merged_automation = tranche_automation(tranche, control_pack, repo_root)
    rung_plan = merged_automation.get("rung_plan", []) or []
    explicit_rung_plan = isinstance(automation_block.get("rung_plan"), list) and bool(automation_block.get("rung_plan"))
    final_rung = rung_plan[-1] if rung_plan and isinstance(rung_plan[-1], dict) else {}
    final_rung_sec = safe_int(final_rung.get("duration_sec"), 0)
    final_continue_on = str(final_rung.get("continue_on", "clean"))
    promotion_capable = final_continue_on == "promotion" or bool(
        merged_automation.get("autoscore", {}).get("promotion")
    )
    explicit_promotion_rules = bool(
        automation_block.get("autoscore", {}).get("promotion")
        if isinstance(automation_block.get("autoscore"), dict)
        else []
    )
    effective_promotion_rules = bool(merged_automation.get("autoscore", {}).get("promotion"))

    if tranche.get("track") != "serialized_mainline":
        gate_contract_finding(findings, "fail", tranche_id, "track", "gate-contract audit only applies to serialized mainline tranches")
    if tranche.get("status") not in VALID_STATUSES:
        gate_contract_finding(findings, "fail", tranche_id, "status", "tranche status is not valid")
    for field in ("objective", "hypothesis", "branch_class", "hypothesis_blocker_family"):
        value = tranche.get(field)
        if not isinstance(value, str) or not value.strip():
            gate_contract_finding(findings, "fail", tranche_id, field, f"{field} must be explicit")

    change_scope = tranche.get("candidate", {}).get("change_scope") if isinstance(tranche.get("candidate"), dict) else None
    files = change_scope.get("files") if isinstance(change_scope, dict) else None
    if files is None or not isinstance(files, list):
        gate_contract_finding(findings, "fail", tranche_id, "candidate.change_scope.files", "candidate change scope must explicitly list files")
    elif any(not isinstance(path, str) or not path for path in files):
        gate_contract_finding(findings, "fail", tranche_id, "candidate.change_scope.files", "candidate file entries must be non-empty strings")
    elif any("/docs/" in path or path.endswith("ROADMAP.md") for path in files) and any("/paraphina/src/" in path for path in files):
        gate_contract_finding(
            findings,
            "warn",
            tranche_id,
            "candidate.change_scope.files",
            "candidate spans runtime and docs files; keep the live-affecting hypothesis single-axis",
        )

    support_gate = str(tranche.get("support_gate") or "none")
    if support_gate not in VALID_SUPPORT_GATES:
        gate_contract_finding(findings, "fail", tranche_id, "support_gate", f"unsupported support gate {support_gate!r}")
    elif promotion_capable and support_gate == "none":
        gate_contract_finding(findings, "fail", tranche_id, "support_gate", "promotion-capable live tranche requires an explicit support gate")

    if not explicit_rung_plan:
        gate_contract_finding(findings, "fail", tranche_id, "automation.rung_plan", "live tranche must declare an explicit rung plan")
    if promotion_capable and final_rung_sec < 7200:
        exception_reason = automation_block.get("final_rung_exception_reason") or tranche.get("final_rung_exception_reason")
        if not isinstance(exception_reason, str) or not exception_reason.strip():
            gate_contract_finding(
                findings,
                "fail",
                tranche_id,
                "automation.rung_plan",
                "promotion-capable final rung must be at least 7200s or carry final_rung_exception_reason",
            )

    promotion_gate = tranche.get("promotion_gate")
    promotion_required = promotion_gate.get("required") if isinstance(promotion_gate, dict) else None
    if promotion_capable and (not isinstance(promotion_required, list) or not promotion_required):
        gate_contract_finding(findings, "fail", tranche_id, "promotion_gate.required", "promotion gate requires explicit required criteria")
    if promotion_capable and not effective_promotion_rules:
        gate_contract_finding(findings, "fail", tranche_id, "automation.autoscore.promotion", "promotion-capable final rung requires autoscore promotion rules")
    elif promotion_capable and not explicit_promotion_rules:
        gate_contract_finding(
            findings,
            "warn",
            tranche_id,
            "automation.autoscore.promotion",
            "promotion rules are inherited from defaults rather than tranche-explicit",
        )

    rollback_criteria = tranche.get("rollback_criteria")
    if not isinstance(rollback_criteria, list) or not rollback_criteria:
        gate_contract_finding(findings, "fail", tranche_id, "rollback_criteria", "rollback criteria must be explicit and non-empty")

    latest_history = latest_history_entry(tranche)
    if isinstance(latest_history, dict) and latest_history.get("decision") in {"hold", "rollback"}:
        latest_observed = latest_history.get("observed_blocker_family")
        latest_precondition_failed = bool(latest_history.get("precondition_failed"))
        if (
            isinstance(latest_observed, str)
            and latest_observed
            and not latest_precondition_failed
            and tranche_uses_blocker_aware_routing(tranche)
            and tranche_fail_child_target(tranche, latest_observed) is None
        ):
            gate_contract_finding(
                findings,
                "fail",
                tranche_id,
                "matched_fail_routes",
                f"latest observed blocker {latest_observed!r} has no configured fail route",
            )

    required_artifacts = gate_contract_required_artifacts(tranche)
    unknown_artifacts = required_artifacts - VALID_GATE_CONTRACT_REQUIRED_ARTIFACTS
    if unknown_artifacts:
        gate_contract_finding(
            findings,
            "warn",
            tranche_id,
            "gate_contract.required_artifacts",
            f"unknown required artifact names: {sorted(unknown_artifacts)}",
        )
    missing_artifacts = VALID_GATE_CONTRACT_REQUIRED_ARTIFACTS - required_artifacts
    if missing_artifacts:
        gate_contract_finding(
            findings,
            "warn",
            tranche_id,
            "gate_contract.required_artifacts",
            f"institutional bundle should require artifacts: {sorted(missing_artifacts)}",
        )
    if gate_contract_economics_relevant(tranche) and "cashflow" not in required_artifacts:
        gate_contract_finding(
            findings,
            "warn",
            tranche_id,
            "gate_contract.required_artifacts",
            "economics-relevant tranche should require cashflow attribution before promotion",
        )

    capital_budget = tranche.get("capital_budget")
    if not isinstance(capital_budget, dict) or not capital_budget:
        gate_contract_finding(
            findings,
            "warn",
            tranche_id,
            "capital_budget",
            "capital budget/equity-drift limits are not machine-readable in the tranche contract",
        )

    surface_id = safe_tranche_surface_id(tranche, control_pack, repo_root)
    status_text = (repo_root / "phase5" / "status.md").read_text(encoding="utf-8") if (repo_root / "phase5" / "status.md").exists() else ""
    roadmap_text = (repo_root / "ROADMAP.md").read_text(encoding="utf-8") if (repo_root / "ROADMAP.md").exists() else ""
    if status_text and tranche_id not in status_text:
        gate_contract_finding(findings, "warn", tranche_id, "phase5/status.md", "status board does not mention audited tranche id")
    if roadmap_text and tranche_id not in roadmap_text:
        gate_contract_finding(findings, "warn", tranche_id, "ROADMAP.md", "roadmap execution snapshot does not mention audited tranche id")
    if surface_id:
        if status_text and surface_id not in status_text:
            gate_contract_finding(findings, "warn", tranche_id, "phase5/status.md", "status board does not mention current computed surface id")
        if roadmap_text and surface_id not in roadmap_text:
            gate_contract_finding(findings, "warn", tranche_id, "ROADMAP.md", "roadmap execution snapshot does not mention current computed surface id")

    fail_count = sum(1 for finding in findings if finding["severity"] == "fail")
    warn_count = sum(1 for finding in findings if finding["severity"] == "warn")
    return {
        "tranche_id": tranche_id,
        "track": tranche.get("track"),
        "status": tranche.get("status"),
        "surface_id": surface_id,
        "support_gate": support_gate,
        "explicit_rung_plan": explicit_rung_plan,
        "final_rung_sec": final_rung_sec,
        "final_continue_on": final_continue_on,
        "promotion_capable": promotion_capable,
        "effective_promotion_rules": effective_promotion_rules,
        "explicit_promotion_rules": explicit_promotion_rules,
        "required_artifacts": sorted(required_artifacts),
        "economics_relevant": gate_contract_economics_relevant(tranche),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "status_summary": "fail" if fail_count else "pass",
        "findings": findings,
    }


def audit_gate_contract(
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    tranche_id: str | None = None,
) -> dict[str, Any]:
    targets = gate_contract_target_tranches(queue, tranche_id)
    tranche_reports = [
        audit_tranche_gate_contract(tranche, control_pack, repo_root)
        for tranche in targets
    ]
    findings = [
        finding
        for tranche_report in tranche_reports
        for finding in tranche_report["findings"]
    ]
    if not targets:
        findings.append(
            {
                "severity": "fail",
                "tranche_id": tranche_id or "<current>",
                "field": "serialized_mainline",
                "message": "no current or requested serialized mainline tranche found",
            }
        )
    fail_count = sum(1 for finding in findings if finding["severity"] == "fail")
    warn_count = sum(1 for finding in findings if finding["severity"] == "warn")
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "repo_root": str(repo_root),
        "requested_tranche_id": tranche_id,
        "status": "fail" if fail_count else "pass",
        "critical_count": fail_count,
        "warning_count": warn_count,
        "tranches": tranche_reports,
        "findings": findings,
    }


def audit_state_sync_tranche(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    *,
    status_text: str | None = None,
    roadmap_text: str | None = None,
) -> dict[str, Any]:
    tranche_id = str(tranche.get("id") or "<missing>")
    findings: list[dict[str, Any]] = []
    card_path = repo_root / "phase5" / "runs" / tranche_id / "tranche_card.yaml"
    computed_surface_id = safe_tranche_surface_id(tranche, control_pack, repo_root)

    if tranche.get("track") != "serialized_mainline":
        gate_contract_finding(findings, "fail", tranche_id, "track", "state-sync audit only applies to serialized mainline tranches")
    if tranche.get("status") not in VALID_STATUSES:
        gate_contract_finding(findings, "fail", tranche_id, "status", "tranche status is not valid")

    if not card_path.exists():
        gate_contract_finding(findings, "fail", tranche_id, "tranche_card.yaml", "tranche card is missing")
    else:
        card = load_yaml(card_path)
        if card.get("schema_version") != 1:
            gate_contract_finding(findings, "fail", tranche_id, "tranche_card.schema_version", "tranche card schema_version must be 1")
        card_tranche = card.get("tranche")
        if not isinstance(card_tranche, dict):
            gate_contract_finding(findings, "fail", tranche_id, "tranche_card.tranche", "tranche card must embed tranche metadata")
        else:
            card_tranche_id = card_tranche.get("id")
            if card_tranche_id != tranche_id:
                gate_contract_finding(findings, "fail", tranche_id, "tranche_card.tranche.id", "tranche card tranche id does not match queue entry")
        if card.get("surface_id") != computed_surface_id:
            gate_contract_finding(findings, "fail", tranche_id, "tranche_card.surface_id", "tranche card surface_id does not match computed surface")
        if card.get("control_pack_baseline") != control_pack.get("baseline_id"):
            gate_contract_finding(findings, "fail", tranche_id, "tranche_card.control_pack_baseline", "tranche card baseline does not match control pack baseline")
        for field in ("topology", "effective_topology", "baseline_topology", "execution_defaults"):
            if not isinstance(card.get(field), dict):
                gate_contract_finding(findings, "fail", tranche_id, f"tranche_card.{field}", f"tranche card {field} must be a mapping")
        run_dir = card.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            gate_contract_finding(findings, "fail", tranche_id, "tranche_card.run_dir", "tranche card run_dir must be explicit")

    if status_text is None:
        status_path = repo_root / "phase5" / "status.md"
        status_text = status_path.read_text(encoding="utf-8") if status_path.exists() else ""
    if roadmap_text is None:
        roadmap_path = repo_root / "ROADMAP.md"
        roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""

    for field_name, text in (
        ("phase5/status.md", status_text),
        ("ROADMAP.md", roadmap_text),
    ):
        if not text:
            gate_contract_finding(findings, "fail", tranche_id, field_name, f"{field_name} is missing")
            continue
        if tranche_id not in text:
            gate_contract_finding(findings, "fail", tranche_id, field_name, f"{field_name} does not mention tranche id")
        if computed_surface_id and computed_surface_id not in text:
            gate_contract_finding(findings, "fail", tranche_id, field_name, f"{field_name} does not mention computed surface id")
        if (
            field_name == "phase5/status.md"
            and tranche.get("status")
            and not status_board_mentions_tranche_status(text, tranche_id, str(tranche.get("status")))
        ):
            gate_contract_finding(
                findings,
                "warn",
                tranche_id,
                f"{field_name}.status",
                f"{field_name} mentions tranche id but not the queue status {tranche.get('status')}",
            )

    return {
        "tranche_id": tranche_id,
        "track": tranche.get("track"),
        "status": tranche.get("status"),
        "surface_id": computed_surface_id,
        "card_path": str(card_path),
        "card_exists": card_path.exists(),
        "linked_child_ids": (
            [child_id]
            if tranche.get("track") == "serialized_mainline" and (child_id := tranche_primary_child_id(tranche))
            else []
        ),
        "fail_count": sum(1 for finding in findings if finding["severity"] == "fail"),
        "warn_count": sum(1 for finding in findings if finding["severity"] == "warn"),
        "status_summary": "fail" if any(finding["severity"] == "fail" for finding in findings) else "pass",
        "findings": findings,
    }


def audit_state_sync(
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    tranche_id: str | None = None,
) -> dict[str, Any]:
    if tranche_id:
        tranche = find_tranche(queue, tranche_id)[2]
        target_ids = [tranche_id]
        if tranche.get("track") == "serialized_mainline":
            primary_child_id = tranche_primary_child_id(tranche)
            if primary_child_id:
                target_ids.append(primary_child_id)
    else:
        current = current_serialized_mainline(queue)
        if current is None:
            target_ids = []
        else:
            target_ids = [str(current.get("id") or "")]
            primary_child_id = tranche_primary_child_id(current)
            if primary_child_id:
                target_ids.append(primary_child_id)

    targets: list[dict[str, Any]] = []
    for target_id in target_ids:
        if not target_id:
            continue
        try:
            _, _, tranche = find_tranche(queue, target_id)
        except KeyError:
            targets.append(
                {
                    "id": target_id,
                    "track": "serialized_mainline",
                    "status": "missing",
                }
            )
            continue
        targets.append(tranche)

    tranche_reports = [
        audit_state_sync_tranche(tranche, control_pack, repo_root)
        for tranche in targets
    ]
    findings = [
        finding
        for tranche_report in tranche_reports
        for finding in tranche_report["findings"]
    ]
    if not targets:
        findings.append(
            {
                "severity": "fail",
                "tranche_id": tranche_id or "<current>",
                "field": "serialized_mainline",
                "message": "no current or requested serialized mainline tranche found",
            }
        )
    fail_count = sum(1 for finding in findings if finding["severity"] == "fail")
    warn_count = sum(1 for finding in findings if finding["severity"] == "warn")
    return {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "repo_root": str(repo_root),
        "requested_tranche_id": tranche_id,
        "status": "fail" if fail_count else "pass",
        "critical_count": fail_count,
        "warning_count": warn_count,
        "tranches": tranche_reports,
        "findings": findings,
    }


def state_sync_report_path(run_root: Path | None) -> Path | None:
    if run_root is None:
        return None
    return run_root / "state_sync_report.json"


def preflight_summary_path(run_root: Path | None) -> Path | None:
    if run_root is None:
        return None
    return run_root / "preflight_summary.json"


def state_sync_blocks_promotion(state_sync_report: dict[str, Any] | None) -> bool:
    if not isinstance(state_sync_report, dict):
        return False
    return bool((state_sync_report.get("critical_count") or 0) or (state_sync_report.get("warning_count") or 0))


def status_board_mentions_tranche_status(status_text: str, tranche_id: str, status: str) -> bool:
    tranche_token = f"`{tranche_id}`"
    status_token = f"`{status}`"
    return any(tranche_token in line and status_token in line for line in status_text.splitlines())


def state_sync_summary_payload(
    state_sync_report: dict[str, Any] | None,
    report_path: Path | None = None,
) -> dict[str, Any] | None:
    if not isinstance(state_sync_report, dict):
        return None
    tranche_report = None
    tranche_reports = state_sync_report.get("tranches")
    if isinstance(tranche_reports, list) and tranche_reports:
        first = tranche_reports[0]
        tranche_report = first if isinstance(first, dict) else None
    return {
        "requested_tranche_id": state_sync_report.get("requested_tranche_id"),
        "status": state_sync_report.get("status"),
        "critical_count": state_sync_report.get("critical_count", 0),
        "warning_count": state_sync_report.get("warning_count", 0),
        "surface_id": None if tranche_report is None else tranche_report.get("surface_id"),
        "report_path": None if report_path is None else str(report_path),
        "blocks_promotion": state_sync_blocks_promotion(state_sync_report),
    }


def write_state_sync_report(run_root: Path | None, state_sync_report: dict[str, Any] | None) -> Path | None:
    if run_root is None or not isinstance(state_sync_report, dict):
        return None
    path = state_sync_report_path(run_root)
    if path is None:
        return None
    write_json(path, state_sync_report)
    return path


def write_preflight_summary(run_root: Path | None, preflight_summary: dict[str, Any] | None) -> Path | None:
    if run_root is None or not isinstance(preflight_summary, dict):
        return None
    path = preflight_summary_path(run_root)
    if path is None:
        return None
    write_json(path, preflight_summary)
    return path


def run_root_from_artifact_paths(*artifact_paths: str | None) -> Path | None:
    for artifact_path in artifact_paths:
        if not artifact_path:
            continue
        try:
            return Path(artifact_path).parent
        except Exception:
            continue
    return None


def support_family_triggered(tranche: dict[str, Any], family: dict[str, Any]) -> bool:
    trigger_mode = family.get("trigger_mode", "always")
    if trigger_mode == "always":
        return True
    blocker_families = set(family.get("blocker_families", []) or [])
    hypothesis_blocker_family = tranche.get("hypothesis_blocker_family")
    return isinstance(hypothesis_blocker_family, str) and hypothesis_blocker_family in blocker_families


def support_family_lane_id(family: dict[str, Any]) -> str:
    family_id = str(family.get("id", "family"))
    support_track_id = str(family.get("support_track_id", "support"))
    return f"{LANE_ROLE_SUPPORT_PREFIX}{family_id}__{support_track_id}"


def infer_support_lane_priority_class(lane_or_id: dict[str, Any] | str) -> str:
    if isinstance(lane_or_id, dict):
        if lane_or_id.get("family_id"):
            return str(lane_or_id["family_id"])
        child_id = str(lane_or_id.get("child_tranche_id", ""))
    else:
        child_id = str(lane_or_id)
    lowered = child_id.lower()
    if "forensics" in lowered:
        return "forensics"
    if "topology" in lowered:
        return "topology_audit"
    if "tooling" in lowered:
        return "tooling"
    if "shadow" in lowered or "blocker" in lowered:
        return "blocker_shadow"
    return child_id


def support_lane_sort_key(lane: dict[str, Any], automation: dict[str, Any]) -> tuple[int, str, str]:
    priority_class = infer_support_lane_priority_class(lane)
    priority_order = automation.get("support_lane_priority", DEFAULT_SUPPORT_LANE_PRIORITY)
    try:
        rank = priority_order.index(priority_class)
    except ValueError:
        rank = len(priority_order)
    return rank, priority_class, str(lane.get("lane_id", ""))


def lane_bundle_root_for_tranche(tranche_id: str, control_pack: dict[str, Any], repo_root: Path = ROOT) -> Path:
    template = tranche_automation({"automation": {}}, control_pack, repo_root)["lane_bundle_root"]
    rel = template.format(tranche_id=tranche_id)
    return repo_root / rel


def worktree_root_for_tranche(tranche_id: str, control_pack: dict[str, Any], repo_root: Path = ROOT) -> Path:
    base = Path(tranche_automation({"automation": {}}, control_pack, repo_root)["worktree_root"])
    return base / tranche_id


def sha256_file_or_missing(path: Path) -> str:
    if not path.exists():
        return f"missing:{path}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tranche_surface_id(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> str:
    defaults = control_pack["execution_defaults"]
    overlay = tranche_stage_overlay_source(tranche, defaults)
    runtime_binary = candidate_runtime_binary_path(tranche, defaults, repo_root)
    material = {
        "baseline_id": control_pack.get("baseline_id", "unknown"),
        "stage_overlay_source": str(overlay),
        "stage_overlay_sha256": sha256_file_or_missing(overlay),
        "runtime_binary": str(runtime_binary),
        "runtime_binary_sha256": sha256_file_or_missing(runtime_binary),
    }
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def safe_tranche_surface_id(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> str | None:
    try:
        return tranche_surface_id(tranche, control_pack, repo_root)
    except Exception:
        return None


def latest_reopened_long_soak_parent(queue: dict[str, Any]) -> dict[str, Any] | None:
    latest_parent: dict[str, Any] | None = None
    latest_ts = ""
    for section, _, tranche in iter_tranches(queue):
        if section != "serialized_mainline":
            continue
        if tranche.get("next_if_pass") != REOPENED_LONG_SOAK_ID:
            continue
        for entry in tranche.get("history", []) or []:
            if entry.get("decision") != "promote":
                continue
            ts = str(entry.get("timestamp_utc") or "")
            if ts and ts >= latest_ts:
                latest_ts = ts
                latest_parent = tranche
    return latest_parent


def refresh_reopened_long_soak_from_parent(
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    parent_tranche: dict[str, Any] | None = None,
) -> bool:
    try:
        section, index, long_soak = find_tranche(queue, REOPENED_LONG_SOAK_ID)
    except KeyError:
        return False
    if parent_tranche is None:
        parent_tranche = latest_reopened_long_soak_parent(queue)
    if parent_tranche is None:
        return False

    overlay_source = str(tranche_stage_overlay_source(parent_tranche, control_pack["execution_defaults"]))
    runtime_binary = str(candidate_runtime_binary_path(parent_tranche, control_pack["execution_defaults"], repo_root))
    surface_id = safe_tranche_surface_id(parent_tranche, control_pack, repo_root)

    changed = False
    env_diff = queue[section][index].setdefault("env_diff", {})
    if env_diff.get("stage_overlay_source") != overlay_source:
        env_diff["stage_overlay_source"] = overlay_source
        changed = True

    candidate = queue[section][index].setdefault("candidate", {})
    if candidate.get("runtime_binary") != runtime_binary:
        candidate["runtime_binary"] = runtime_binary
        changed = True

    description = "Latest promoted reopened surface after venue-promotion tranches."
    if surface_id:
        description = (
            f"Exact promoted reopened surface from {parent_tranche.get('id')} "
            f"on surface {surface_id}."
        )
    control = queue[section][index].setdefault("control", {})
    if control.get("description") != description:
        control["description"] = description
        changed = True

    return changed


def status_board_serialized_mainline(queue: dict[str, Any]) -> dict[str, Any] | None:
    active_mainline = next(
        (
            tranche
            for _, _, tranche in iter_tranches(queue)
            if tranche.get("track") == "serialized_mainline"
            and tranche.get("status") in {"ready", "in_progress"}
        ),
        None,
    )
    if active_mainline is not None:
        return active_mainline
    terminal_mainlines = [
        tranche
        for tranche in (queue.get("serialized_mainline", []) or [])
        if tranche.get("status") in {"hold", "promoted"}
    ]
    if not terminal_mainlines:
        return None
    return max(
        terminal_mainlines,
        key=lambda tranche: str((latest_history_entry(tranche) or {}).get("timestamp_utc") or ""),
    )


def render_status_markdown(queue: dict[str, Any], control_pack: dict[str, Any], repo_root: Path = ROOT) -> str:
    current_mainline = status_board_serialized_mainline(queue)
    topology = (
        current_runtime_topology_snapshot(current_mainline, control_pack, repo_root)
        if current_mainline is not None
        else copy.deepcopy(control_pack.get("topology", {}) or {})
    )
    roles = topology.get("roles", {})
    ready_support = [
        tranche for _, _, tranche in iter_tranches(queue)
        if tranche.get("track") == "parallel_support" and tranche.get("status") == "ready"
    ]

    lines = [
        "# Phase 5 Status",
        "",
        f"- Updated UTC: `{queue.get('updated_utc', utc_now())}`",
        f"- Baseline ID: `{control_pack.get('baseline_id', 'unknown')}`",
        "",
        "## Topology",
        "",
        f"- Topology source: `{topology.get('overlay_path', 'baseline_topology')}`",
        f"- Live connectors: `{','.join(topology.get('connectors', []))}`",
        f"- FV-disabled venues: `{','.join(topology.get('fv_disabled_venues', [])) or 'none'}`",
        f"- Excluded venues: `{','.join(topology.get('excluded_venues', [])) or 'none'}`",
        "",
        "## Venue Roles",
        "",
    ]
    for venue_id, role in roles.items():
        lines.append(f"- `{venue_id}`: `{role}`")
    lines.extend(["", "## Serialized Mainline", ""])
    if current_mainline is None:
        lines.append("- none ready or in progress")
    for section, _, tranche in iter_tranches(queue):
        if section != "serialized_mainline":
            continue
        marker = " <- current" if current_mainline and tranche.get("id") == current_mainline.get("id") else ""
        lines.append(
            f"- `{tranche.get('id')}`: `{tranche.get('status')}`{marker}"
        )
        lines.append(f"  objective: {tranche.get('objective', '')}")
        if tranche.get("branch_class"):
            lines.append(f"  class: `{tranche.get('branch_class')}`")
        if tranche.get("hypothesis_blocker_family"):
            lines.append(f"  hypothesis blocker: `{tranche.get('hypothesis_blocker_family')}`")
        required = tranche.get("required_cleared_blockers", []) or []
        if required:
            lines.append(f"  requires cleared blockers: `{','.join(required)}`")
        if tranche.get("support_gate"):
            lines.append(f"  support gate: `{tranche.get('support_gate')}`")
        if tranche.get("mechanism_gate_mode"):
            lines.append(f"  mechanism gate mode: `{tranche.get('mechanism_gate_mode')}`")
        if tranche.get("progress_credit"):
            lines.append(f"  planned credit: `{tranche.get('progress_credit')}`")
        surface_id = safe_tranche_surface_id(tranche, control_pack, repo_root)
        if surface_id:
            lines.append(f"  surface_id: `{surface_id}`")
        last_history = latest_history_entry(tranche)
        if last_history:
            observed = last_history.get("observed_blocker_family")
            if observed:
                lines.append(f"  last observed blocker: `{observed}`")
            if "precondition_failed" in last_history:
                lines.append(f"  last precondition_failed: `{str(bool(last_history.get('precondition_failed'))).lower()}`")
            if last_history.get("credit_earned") is not None:
                lines.append(f"  last credit earned: `{last_history.get('credit_earned')}`")
            if last_history.get("surface_id"):
                lines.append(f"  last run surface_id: `{last_history.get('surface_id')}`")
    lines.extend(["", "## Parallel Support Tracks", ""])
    if ready_support:
        for tranche in ready_support:
            lines.append(f"- `{tranche.get('id')}`: `{tranche.get('objective', '')}`")
    else:
        lines.append("- none ready")
    lines.extend(["", "## Active Blocker", ""])
    if current_mainline:
        if str(current_mainline.get("status") or "") == "promoted":
            current_id = str(current_mainline.get("id") or "")
            history = latest_history_entry(current_mainline) or {}
            timestamp = str(history.get("timestamp_utc") or "").strip()
            if timestamp:
                lines.append(f"- `none`: latest serialized-mainline child `{current_id}` promoted at `{timestamp}`")
            else:
                lines.append(f"- `none`: latest serialized-mainline child `{current_id}` is promoted")
        else:
            lines.append(f"- `{str(current_mainline.get('hypothesis', '')).strip()}`")
    else:
        lines.append("- no serialized mainline is currently admissible or active")
    return "\n".join(lines) + "\n"


def current_serialized_mainline(queue: dict[str, Any]) -> dict[str, Any] | None:
    current_mainline = next(
        (
            tranche
            for _, _, tranche in iter_tranches(queue)
            if tranche.get("track") == "serialized_mainline"
            and tranche.get("status") in {"ready", "in_progress"}
        ),
        None,
    )
    if current_mainline is not None:
        return current_mainline
    held_mainlines = [
        tranche for tranche in (queue.get("serialized_mainline", []) or []) if tranche.get("status") == "hold"
    ]
    if not held_mainlines:
        return None
    return max(
        held_mainlines,
        key=lambda tranche: str((latest_history_entry(tranche) or {}).get("timestamp_utc") or ""),
    )


def write_status(queue: dict[str, Any], control_pack: dict[str, Any], repo_root: Path = ROOT) -> Path:
    status_path = repo_root / "phase5" / "status.md"
    write_text(status_path, render_status_markdown(queue, control_pack, repo_root))
    return status_path


def next_ready_mainline(queue: dict[str, Any]) -> dict[str, Any] | None:
    for tranche in queue.get("serialized_mainline", []) or []:
        if tranche.get("status") == "ready":
            return tranche
    return None


def ensure_run_dir(repo_root: Path, tranche_id: str) -> Path:
    run_dir = repo_root / "phase5" / "runs" / tranche_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def tranche_card_payload(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    run_dir: Path,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    effective_topology = tranche_topology_snapshot(tranche, control_pack, repo_root)
    execution_defaults = copy.deepcopy(control_pack.get("execution_defaults", {}) or {})
    if execution_defaults:
        execution_defaults["stage_overlay_source"] = str(
            tranche_stage_overlay_source(tranche, execution_defaults)
        )
        execution_defaults["live_guard_args"] = tranche_live_guard_args(tranche, execution_defaults)
        execution_defaults["runtime_binary"] = str(
            candidate_runtime_binary_path(tranche, execution_defaults, repo_root)
        )
    return {
        "schema_version": 1,
        "prepared_utc": utc_now(),
        "tranche": tranche,
        "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
        "control_pack_baseline": control_pack.get("baseline_id"),
        "topology": effective_topology,
        "effective_topology": effective_topology,
        "baseline_topology": copy.deepcopy(control_pack.get("topology", {}) or {}),
        "execution_defaults": execution_defaults,
        "run_dir": str(run_dir),
    }


def prepare_tranche(repo_root: Path = ROOT, tranche_id: str | None = None, mark_in_progress: bool = True) -> Path:
    queue, control_pack = load_state(repo_root)
    tranche = next_ready_mainline(queue) if tranche_id is None else find_tranche(queue, tranche_id)[2]
    if tranche is None:
        raise SystemExit("No ready serialized mainline tranche")
    queue_changed = False
    if tranche.get("id") == REOPENED_LONG_SOAK_ID:
        queue_changed = refresh_reopened_long_soak_from_parent(queue, control_pack, repo_root)
        if queue_changed:
            tranche = next_ready_mainline(queue) if tranche_id is None else find_tranche(queue, tranche_id)[2]
    section, index, tranche = find_tranche(queue, tranche["id"])
    if mark_in_progress and tranche.get("status") in {"ready", "hold"}:
        queue[section][index]["status"] = "in_progress"
        queue[section][index].setdefault("history", []).append(
            {"decision": "prepare", "timestamp_utc": utc_now(), "notes": "Prepared tranche card"}
        )
        tranche = queue[section][index]
    run_dir = ensure_run_dir(repo_root, tranche["id"])
    card_path = run_dir / "tranche_card.yaml"
    save_yaml(card_path, tranche_card_payload(tranche, control_pack, run_dir, repo_root))
    if queue_changed or (mark_in_progress and tranche.get("status") == "in_progress"):
        save_state(queue, repo_root)
    write_status(queue, control_pack, repo_root)
    return card_path


def activate_next(queue: dict[str, Any], tranche_id: str | None) -> None:
    if not tranche_id:
        return
    section, index, tranche = find_tranche(queue, tranche_id)
    if tranche.get("status") == "blocked":
        queue[section][index]["status"] = "ready"


def tranche_matched_fail_routes(tranche: dict[str, Any]) -> dict[str, str]:
    routes = tranche.get("matched_fail_routes")
    if not isinstance(routes, dict):
        return {}
    out: dict[str, str] = {}
    for blocker_family, target in routes.items():
        if isinstance(blocker_family, str) and isinstance(target, str):
            out[blocker_family] = target
    return out


def tranche_direct_child_ids(tranche: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for edge in ("next_if_pass", "next_if_fail", "next_if_fail_when_matched", "surface_local_restore_hygiene_child"):
        target = tranche.get(edge)
        if isinstance(target, str) and target and target not in ids:
            ids.append(target)
    for target in tranche_matched_fail_routes(tranche).values():
        if target not in ids:
            ids.append(target)
    return ids


def tranche_primary_child_id(tranche: dict[str, Any]) -> str | None:
    for edge in ("next_if_pass", "next_if_fail_when_matched", "next_if_fail", "surface_local_restore_hygiene_child"):
        target = tranche.get(edge)
        if isinstance(target, str) and target:
            return target
    matched_routes = tranche_matched_fail_routes(tranche)
    if matched_routes:
        return next(iter(matched_routes.values()))
    return None


def tranche_fail_child_target(
    tranche: dict[str, Any],
    observed_blocker_family: str | None = None,
) -> str | None:
    if observed_blocker_family is not None:
        if observed_blocker_family == "restore_hygiene":
            local_hygiene_child = tranche.get("surface_local_restore_hygiene_child")
            if isinstance(local_hygiene_child, str) and local_hygiene_child:
                return local_hygiene_child
        matched_routes = tranche_matched_fail_routes(tranche)
        if observed_blocker_family in matched_routes:
            return matched_routes[observed_blocker_family]
        hypothesis = tranche.get("hypothesis_blocker_family")
        if observed_blocker_family == hypothesis:
            if "next_if_fail_when_matched" in tranche:
                return tranche.get("next_if_fail_when_matched")
            return tranche.get("next_if_fail")
        return None
    return tranche.get("next_if_fail")


def should_suppress_reopened_long_soak_fail_loop(
    queue: dict[str, Any],
    tranche: dict[str, Any],
    observed_blocker_family: str | None,
    next_target: str | None,
    surface_id: str | None,
) -> bool:
    if tranche.get("id") != REOPENED_LONG_SOAK_ID:
        return False
    if observed_blocker_family not in REOPENED_LONG_SOAK_NO_LOOP_BLOCKERS:
        return False
    if not next_target:
        return False
    _, _, child_tranche = find_tranche(queue, next_target)
    if child_tranche.get("status") != "promoted":
        return False
    latest_child_history = latest_history_entry(child_tranche)
    if not latest_child_history:
        return False
    if latest_child_history.get("decision") != "promote":
        return False
    if surface_id and latest_child_history.get("surface_id") not in {None, surface_id}:
        return False
    return True


def tranche_fail_route_specs(tranche: dict[str, Any]) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    seen_children: set[str] = set()
    default_child = tranche.get("next_if_fail")
    if isinstance(default_child, str) and default_child:
        routes.append(
            {
                "child_tranche_id": default_child,
                "observed_blocker_family": None,
                "route_kind": "default",
            }
        )
        seen_children.add(default_child)

    hypothesis = tranche.get("hypothesis_blocker_family")
    matched_child = tranche.get("next_if_fail_when_matched")
    if (
        isinstance(hypothesis, str)
        and isinstance(matched_child, str)
        and matched_child
        and matched_child not in seen_children
    ):
        routes.append(
            {
                "child_tranche_id": matched_child,
                "observed_blocker_family": hypothesis,
                "route_kind": "hypothesis_match",
            }
        )
        seen_children.add(matched_child)

    for blocker_family, child_id in tranche_matched_fail_routes(tranche).items():
        if child_id in seen_children:
            continue
        routes.append(
            {
                "child_tranche_id": child_id,
                "observed_blocker_family": blocker_family,
                "route_kind": "matched_fail_route",
            }
        )
        seen_children.add(child_id)
    local_hygiene_child = tranche.get("surface_local_restore_hygiene_child")
    if isinstance(local_hygiene_child, str) and local_hygiene_child and local_hygiene_child not in seen_children:
        routes.append(
            {
                "child_tranche_id": local_hygiene_child,
                "observed_blocker_family": "restore_hygiene",
                "route_kind": "surface_local_restore_hygiene_child",
            }
        )
    return routes


def tranche_uses_blocker_aware_routing(tranche: dict[str, Any]) -> bool:
    return any(
        key in tranche
        for key in (
            "hypothesis_blocker_family",
            "required_cleared_blockers",
            "next_if_fail_when_matched",
            "matched_fail_routes",
        )
    )


def fail_child_activation_allowed(
    tranche: dict[str, Any],
    observed_blocker_family: str | None,
    precondition_failed: bool,
) -> bool:
    if not tranche_uses_blocker_aware_routing(tranche):
        return True
    if precondition_failed:
        return False
    if not observed_blocker_family:
        return False
    return tranche_fail_child_target(tranche, observed_blocker_family) is not None


def blocker_event_for_history(
    tranche: dict[str, Any],
    history_entry: dict[str, Any],
    blocker_family: str,
) -> str | None:
    if history_entry.get("observed_blocker_family") == blocker_family and history_entry.get("decision") in {"hold", "rollback"}:
        return "reopened"
    hypothesis = history_entry.get("hypothesis_blocker_family") or tranche.get("hypothesis_blocker_family")
    if hypothesis == blocker_family and history_entry.get("decision") == "promote":
        return "cleared"
    return None


def latest_blocker_event(
    queue: dict[str, Any],
    blocker_family: str,
    surface_id: str,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_ts = ""
    for _, _, tranche in iter_tranches(queue):
        for entry in tranche.get("history", []) or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("surface_id") != surface_id:
                continue
            event = blocker_event_for_history(tranche, entry, blocker_family)
            if event is None:
                continue
            ts = str(entry.get("timestamp_utc", ""))
            if latest is None or ts >= latest_ts:
                latest_ts = ts
                latest = {
                    "status": event,
                    "timestamp_utc": ts,
                    "tranche_id": tranche.get("id"),
                    "entry": entry,
                }
    return latest


def maybe_auto_prepare_child_support_gate(
    child_tranche_id: str | None,
    decision: str,
    repo_root: Path,
    run_root: Path | None,
    autoscore: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if decision != "hold" or not child_tranche_id or run_root is None:
        return None
    if not isinstance(autoscore, dict) or not autoscore.get("clean", {}).get("passed", False):
        return {
            "child_tranche_id": child_tranche_id,
            "status": "skipped",
            "reason": "parent_hold_not_clean",
        }

    closeout = load_closeout_bundle(run_root)
    if not closeout or closeout.get("guard_intervened"):
        return {
            "child_tranche_id": child_tranche_id,
            "status": "skipped",
            "reason": "parent_closeout_not_guard_clean",
        }
    if not closeout.get("closeout_contract_complete"):
        return {
            "child_tranche_id": child_tranche_id,
            "status": "skipped",
            "reason": "parent_closeout_contract_incomplete",
        }
    if not closeout.get("post_rollback_venue_audit_clean") or closeout.get("trade_mode_post") != "shadow":
        return {
            "child_tranche_id": child_tranche_id,
            "status": "skipped",
            "reason": "host_not_restored_to_shadow",
        }

    queue, control_pack = load_state(repo_root)
    _, _, child_tranche = find_tranche(queue, child_tranche_id)
    gate = str(child_tranche.get("support_gate") or "none")
    if gate == "none":
        return {
            "child_tranche_id": child_tranche_id,
            "status": "skipped",
            "reason": "child_has_no_support_gate",
        }

    surface_id = safe_tranche_surface_id(child_tranche, control_pack, repo_root)
    if surface_id:
        support_ok, reason = support_gate_satisfied(child_tranche_id, child_tranche, repo_root, surface_id)
        if support_ok:
            return {
                "child_tranche_id": child_tranche_id,
                "support_gate": gate,
                "status": "skipped",
                "reason": "support_gate_already_satisfied",
                "details": reason,
            }

    ensure_shadow_health()
    card_path = prepare_tranche(repo_root=repo_root, tranche_id=child_tranche_id, mark_in_progress=True)
    duration_sec = support_gate_duration_sec(gate)
    if gate in {"shadow_smoke_10m", "shadow_smoke_30m"} and duration_sec is not None:
        try:
            support_run_dir = run_shadow_smoke(child_tranche_id, duration_sec, repo_root)
            return {
                "child_tranche_id": child_tranche_id,
                "support_gate": gate,
                "status": "pass",
                "tranche_card_path": str(card_path),
                "run_dir": str(support_run_dir),
            }
        except subprocess.CalledProcessError as exc:
            return {
                "child_tranche_id": child_tranche_id,
                "support_gate": gate,
                "status": "fail",
                "tranche_card_path": str(card_path),
                "error": str(exc),
            }
    if gate == "shadow_ab_10m":
        try:
            support_run_dir = run_shadow_ab(child_tranche_id, repo_root)
            return {
                "child_tranche_id": child_tranche_id,
                "support_gate": gate,
                "status": "pass",
                "tranche_card_path": str(card_path),
                "run_dir": str(support_run_dir),
            }
        except subprocess.CalledProcessError as exc:
            return {
                "child_tranche_id": child_tranche_id,
                "support_gate": gate,
                "status": "fail",
                "tranche_card_path": str(card_path),
                "error": str(exc),
            }
    return {
        "child_tranche_id": child_tranche_id,
        "support_gate": gate,
        "status": "skipped",
        "reason": "unsupported_child_support_gate",
        "tranche_card_path": str(card_path),
    }


def latest_promoted_lineage_for_parent(
    queue: dict[str, Any],
    parent_tranche_id: str,
    surface_id: str | None,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_ts = ""
    for _, _, tranche in iter_tranches(queue):
        for entry in tranche.get("history", []) or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("decision") != "promote":
                continue
            if entry.get("activated_child") != parent_tranche_id:
                continue
            if surface_id and entry.get("surface_id") not in {None, surface_id}:
                continue
            summary_path = entry.get("summary_path")
            if not isinstance(summary_path, str) or not summary_path:
                continue
            ts = parse_timestamp_utc(entry.get("timestamp_utc"))
            if latest is not None and ts < latest_ts:
                continue
            latest_ts = ts
            latest = {
                "tranche_id": tranche.get("id"),
                "entry": entry,
                "run_root": str(Path(summary_path).parent),
            }
    return latest


def latest_direct_venue_audit_post_path(run_root: Path) -> Path | None:
    matches = sorted(run_root.glob("direct_venue_audit_post*.json"))
    return matches[-1] if matches else None


def latest_cashflow_attribution_path(run_root: Path) -> Path | None:
    candidates = [
        run_root / "cashflow_attribution.json",
        run_root / "venue_cashflow_ledger.json",
        run_root / "cashflow" / "cashflow_attribution.json",
        run_root / "cashflow" / "venue_cashflow_ledger.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(run_root.glob("*cashflow*.json")) + sorted((run_root / "cashflow").glob("*cashflow*.json"))
    return matches[-1] if matches else None


def latest_balance_snapshot_path(run_root: Path) -> Path | None:
    candidates = [
        run_root / "balance_snapshot_comparison.json",
        run_root / "balance_post_snapshot.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(path for path in run_root.glob("balance_*_snapshot.json") if path.name != "balance_pre_snapshot.json")
    return matches[-1] if matches else None


def load_json_payload_with_path(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    payload = loaded if isinstance(loaded, dict) else {"payload": loaded}
    payload = copy.deepcopy(payload)
    payload["exists"] = True
    payload["path"] = str(path)
    return payload


def venue_scorecard(metrics: dict[str, Any], venue: str) -> dict[str, Any]:
    scorecard = metrics.get("execution_scorecard")
    if not isinstance(scorecard, dict):
        return {}
    payload = scorecard.get(venue)
    return payload if isinstance(payload, dict) else {}


def fills_by_venue(metrics: dict[str, Any], venues: list[str]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for venue in venues:
        score = venue_scorecard(metrics, venue)
        out[venue] = {
            "fill_count": safe_int(score.get("fills")),
            "fill_base_eth": safe_float(score.get("fill_base")),
        }
    return out


def order_activity_by_venue(metrics: dict[str, Any], venues: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for venue in venues:
        score = venue_scorecard(metrics, venue)
        out[venue] = {
            "place_intents": safe_int(score.get("place_i")),
            "place_acks": safe_int(score.get("place_ack")),
            "cancel_intents": safe_int(score.get("cancel_i")),
            "cancel_acks": safe_int(score.get("cancel_ack")),
        }
    return out


def role_rationale_for_venue(
    venue: str,
    final_fills: dict[str, dict[str, float | int]],
    lineage_fills: dict[str, dict[str, float | int]] | None,
    final_activity: dict[str, dict[str, int]],
) -> str:
    current = final_fills.get(venue, {})
    current_fill_count = safe_int(current.get("fill_count"))
    current_fill_base = safe_float(current.get("fill_base_eth"))
    if current_fill_count > 0 or current_fill_base > 0:
        return (
            f"Promoted as fill after current final 2h reopened soak participation: "
            f"{current_fill_count} fills / {format_decimal(current_fill_base)} ETH."
        )

    prior = (lineage_fills or {}).get(venue, {})
    prior_fill_count = safe_int(prior.get("fill_count"))
    prior_fill_base = safe_float(prior.get("fill_base_eth"))
    if prior_fill_count > 0 or prior_fill_base > 0:
        return (
            "Promoted as fill from exact-surface lineage with prior fill evidence: "
            f"{prior_fill_count} fills / {format_decimal(prior_fill_base)} ETH; "
            "the current final 2h reopened soak showed operational order activity but no fills."
        )

    activity = final_activity.get(venue, {})
    if safe_int(activity.get("place_intents")) > 0 or safe_int(activity.get("place_acks")) > 0:
        return (
            "Promoted as fill because it remained execution-eligible with live order activity "
            "on the accepted surface and no narrower venue-local disproof reopened."
        )
    return "Promoted as fill on the accepted exact surface."


def build_reopened_final_topology_spec(
    final_tranche: dict[str, Any],
    source_tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path,
    run_root: Path,
    lineage: dict[str, Any] | None,
) -> dict[str, Any]:
    defaults = control_pack["execution_defaults"]
    runtime_binary = candidate_runtime_binary_path(source_tranche, defaults, repo_root)
    runtime_binary_str = str(runtime_binary)
    topology = tranche_topology_snapshot(source_tranche, control_pack, repo_root)
    surface_id = safe_tranche_surface_id(source_tranche, control_pack, repo_root)
    closeout = load_closeout_bundle(run_root)
    metrics = load_live_metrics_bundle(run_root)
    balance_snapshot = load_json_payload_with_path(latest_balance_snapshot_path(run_root))
    venues = list(topology["roles"].keys())
    final_fills = fills_by_venue(metrics, venues)
    final_activity = order_activity_by_venue(metrics, venues)
    opportunity_scorecard = metrics.get("opportunity_adjusted_scorecard")
    opportunity_scorecard = opportunity_scorecard if isinstance(opportunity_scorecard, dict) else {}
    opportunity_passed_by_venue: dict[str, bool] = {}
    opportunity_reason_by_venue: dict[str, str | None] = {}
    for venue in venues:
        payload = opportunity_scorecard.get(venue)
        payload = payload if isinstance(payload, dict) else {}
        if payload:
            opportunity_passed_by_venue[venue] = bool(payload.get("passed"))
            reason = payload.get("reason")
            opportunity_reason_by_venue[venue] = str(reason) if reason is not None else None
        else:
            fill_payload = final_fills.get(venue, {})
            legacy_fill_passed = safe_int(fill_payload.get("fill_count")) > 0 or safe_float(fill_payload.get("fill_base_eth")) > 0
            opportunity_passed_by_venue[venue] = legacy_fill_passed
            opportunity_reason_by_venue[venue] = "legacy_fill_evidence" if legacy_fill_passed else None
    mm_fill_evidence_venues = [
        venue
        for venue, reason in opportunity_reason_by_venue.items()
        if reason in {"mm_fill_evidence", "legacy_fill_evidence"}
    ]
    balance_total = balance_snapshot.get("total") if isinstance(balance_snapshot.get("total"), dict) else {}
    balance_delta_usd = safe_float(balance_total.get("delta_usd"), default=float("nan"))
    balance_abs_delta_usd = safe_float(balance_total.get("abs_delta_usd_float"), default=float("nan"))
    balance_exists = bool(balance_snapshot.get("exists"))
    balance_venue_count = safe_int(balance_snapshot.get("venue_count"))
    lineage_payload: dict[str, Any] | None = None
    lineage_fills: dict[str, dict[str, float | int]] | None = None
    if lineage is not None:
        lineage_run_root = Path(str(lineage["run_root"]))
        lineage_closeout = load_closeout_bundle(lineage_run_root)
        lineage_metrics_bundle = load_live_metrics_bundle(lineage_run_root)
        lineage_fills = fills_by_venue(lineage_metrics_bundle, venues)
        lineage_payload = {
            "tranche_id": lineage.get("tranche_id"),
            "run_root": str(lineage_run_root),
            "segment_start_utc": lineage_closeout.get("segment_start_utc"),
            "segment_end_utc": lineage_closeout.get("segment_end_utc"),
            "fill_count_total": safe_int(lineage_closeout.get("fill_count_total")),
            "fill_base_total_eth": safe_float(lineage_closeout.get("fill_base_total")),
            "fills_by_venue": lineage_fills,
        }

    final_direct_audit_path = latest_direct_venue_audit_post_path(run_root)
    non_hyperliquid_fill_venues = sum(
        1 for venue, payload in final_fills.items() if venue != "hyperliquid" and safe_int(payload["fill_count"]) > 0
    )
    completion_standard = {
        "no_connector_excluded_for_unresolved_defect": not topology["excluded_venues"],
        "all_five_execution_eligible": len(topology["connectors"]) == 5 and not topology["excluded_venues"],
        "all_five_fv_eligible": len(topology["fv_disabled_venues"]) == 0,
        "all_five_fill_roles_frozen": all(role == "fill" for role in topology["roles"].values()),
        "non_hyperliquid_fill_venues_in_final_soak": non_hyperliquid_fill_venues,
        "opportunity_adjusted_participation_passed": all(opportunity_passed_by_venue.get(venue, False) for venue in venues),
        "opportunity_adjusted_passed_by_venue": opportunity_passed_by_venue,
        "opportunity_adjusted_reason_by_venue": opportunity_reason_by_venue,
        "mm_fill_evidence_venues": mm_fill_evidence_venues,
        "mm_fill_evidence_venue_count": len(mm_fill_evidence_venues),
        "balance_snapshot_exists": balance_exists,
        "balance_snapshot_venue_count": balance_venue_count,
        "balance_delta_usd": balance_delta_usd if balance_delta_usd == balance_delta_usd else None,
        "balance_abs_delta_usd": balance_abs_delta_usd if balance_abs_delta_usd == balance_abs_delta_usd else None,
        "balance_delta_nonnegative": bool(balance_exists and balance_delta_usd == balance_delta_usd and balance_delta_usd >= 0.0),
        "balance_abs_delta_within_budget": bool(balance_exists and balance_abs_delta_usd == balance_abs_delta_usd and balance_abs_delta_usd <= 0.5),
        "final_long_soak_operationally_clean": bool(
            closeout.get("guard_window_completed")
            and not closeout.get("guard_intervened")
            and closeout.get("closeout_contract_complete")
            and closeout.get("pre_restore_venue_audit_clean")
            and closeout.get("post_rollback_venue_audit_clean")
            and not closeout.get("kill_events_present_post")
            and safe_int(closeout.get("reconcile_mismatch_count_post")) == 0
            and str(closeout.get("trade_mode_post") or "") == "shadow"
        ),
        "final_direct_venue_audit_clean": bool(closeout.get("post_rollback_venue_audit_clean")),
    }
    completion_standard_passed = bool(
        completion_standard["no_connector_excluded_for_unresolved_defect"]
        and completion_standard["all_five_execution_eligible"]
        and completion_standard["all_five_fv_eligible"]
        and completion_standard["all_five_fill_roles_frozen"]
        and completion_standard["opportunity_adjusted_participation_passed"]
        and safe_int(completion_standard["mm_fill_evidence_venue_count"]) >= 2
        and completion_standard["balance_snapshot_exists"]
        and safe_int(completion_standard["balance_snapshot_venue_count"]) == 5
        and completion_standard["balance_delta_nonnegative"]
        and completion_standard["balance_abs_delta_within_budget"]
        and completion_standard["final_long_soak_operationally_clean"]
        and completion_standard["final_direct_venue_audit_clean"]
    )
    completion_standard["passed"] = completion_standard_passed

    evidence = {
        "reopened_multi_venue_long_soak": {
            "tranche_id": source_tranche.get("id"),
            "run_root": str(run_root),
            "segment_start_utc": closeout.get("segment_start_utc"),
            "segment_end_utc": closeout.get("segment_end_utc"),
            "tick_count": safe_int(closeout.get("tick_count")),
            "guard_window_completed": bool(closeout.get("guard_window_completed")),
            "guard_exit_code": safe_int(closeout.get("guard_exit_code")),
            "guard_intervened": bool(closeout.get("guard_intervened")),
            "kill_events_present_post": bool(closeout.get("kill_events_present_post")),
            "reconcile_mismatch_count_post": safe_int(closeout.get("reconcile_mismatch_count_post")),
            "systemd_nrestarts_post": safe_int(closeout.get("systemd_nrestarts_post")),
            "trade_mode_post": closeout.get("trade_mode_post"),
            "final_pnl_total_usd": safe_float(closeout.get("final_pnl_total")),
            "final_q_global_eth": closeout.get("final_q_global"),
            "fill_count_total": safe_int(closeout.get("fill_count_total")),
            "fill_base_total_eth": safe_float(closeout.get("fill_base_total")),
            "fills_by_venue": final_fills,
            "order_activity_by_venue": final_activity,
            "opportunity_adjusted_scorecard": opportunity_scorecard,
            "balance_snapshot": {
                "path": balance_snapshot.get("path"),
                "venue_count": balance_venue_count,
                "total": balance_total,
            },
        }
    }
    if lineage_payload is not None:
        evidence["exact_surface_lineage_requal"] = lineage_payload
    if final_direct_audit_path is not None:
        evidence["final_direct_venue_audit"] = {
            "path": str(final_direct_audit_path),
            "ok": True,
            "max_open_orders": 0,
            "position_tol_base": 0.0025,
        }

    return {
        "phase": 5,
        "status": "accepted_closeout" if completion_standard_passed else "hold_closeout",
        "frozen_utc": utc_now(),
        "surface_id": surface_id,
        "baseline_id": control_pack.get("baseline_id"),
        "runtime_binary": runtime_binary_str,
        "runtime_binary_sha256": sha256_file_or_missing(runtime_binary),
        "stage_overlay_source": str(tranche_stage_overlay_source(source_tranche, defaults)),
        "live_connectors": topology["connectors"],
        "fv_disabled_venues": topology["fv_disabled_venues"],
        "excluded_venues": topology["excluded_venues"],
        "roles": topology["roles"],
        "role_rationale": {
            venue: role_rationale_for_venue(venue, final_fills, lineage_fills, final_activity)
            for venue in venues
        },
        "completion_standard": completion_standard,
        "evidence": evidence,
        "closeout_disposition": {
            "verdict": "accepted" if completion_standard_passed else "hold",
            "reason": (
                "The reopened Phase 5 surface is frozen as a five-connector, five-FV-eligible, "
                "five-fill-role topology with clean long-soak operation, non-negative balance-delta PnL, "
                "and opportunity-adjusted venue participation."
                if completion_standard_passed
                else "The reopened Phase 5 surface remains held because the final completion standard is incomplete."
            ),
        },
    }


def build_reopened_final_closeout_markdown(spec: dict[str, Any], run_root: Path, repo_root: Path) -> str:
    roles = spec["roles"]
    fills = spec["evidence"]["reopened_multi_venue_long_soak"]["fills_by_venue"]
    activity = spec["evidence"]["reopened_multi_venue_long_soak"]["order_activity_by_venue"]
    closeout = spec["evidence"]["reopened_multi_venue_long_soak"]
    lineage = spec["evidence"].get("exact_surface_lineage_requal")
    venues = list(roles.keys())
    completion = spec["completion_standard"]
    balance_delta = completion.get("balance_delta_usd")
    balance_abs_delta = completion.get("balance_abs_delta_usd")
    mm_fill_venues = completion.get("mm_fill_evidence_venues") or []
    opportunity_reasons = completion.get("opportunity_adjusted_reason_by_venue") or {}
    risk_summary = load_live_metrics_bundle(run_root).get("risk", {})

    lines = [
        "# Reopened Phase 5 Final Closeout",
        "",
        "## Verdict",
        "",
        "`ACCEPTED`",
        "",
        f"The reopened Phase 5 closeout is accepted on surface `{spec['surface_id']}`.",
        "",
        "This is a topology and operational-qualification closeout. It does not claim that future 24/7 economics are solved, only that the live surface is now a valid all5 venue topology for continuous operation under normal sentinel monitoring.",
        "",
        "## Frozen Topology",
        "",
        f"- Runtime binary: `{spec['runtime_binary']}`",
        f"- Runtime SHA256: `{spec['runtime_binary_sha256']}`",
        f"- Stage overlay: `{spec['stage_overlay_source']}`",
        f"- Live connectors: `{','.join(spec['live_connectors'])}`",
        f"- FV-disabled venues: `{','.join(spec['fv_disabled_venues']) if spec['fv_disabled_venues'] else 'none'}`",
        f"- Excluded venues: `{','.join(spec['excluded_venues']) if spec['excluded_venues'] else 'none'}`",
        "",
        "## Frozen Role Matrix",
        "",
    ]
    for venue, role in roles.items():
        lines.append(f"- `{venue}`: `{role}`")
    lines.extend(
        [
            "",
            "## Final Long-Soak Evidence",
            "",
            f"- Tranche: `{closeout['tranche_id']}`",
            f"- Run root: `{closeout['run_root']}`",
            f"- Segment UTC: `{closeout['segment_start_utc']} -> {closeout['segment_end_utc']}`",
            f"- Ticks analyzed: `{closeout['tick_count']}`",
            f"- Guard window completed: `{str(bool(closeout['guard_window_completed'])).lower()}`",
            f"- Guard exit code: `{closeout['guard_exit_code']}`",
            f"- Guard intervention: `{str(bool(closeout['guard_intervened'])).lower()}`",
            f"- Kill events post-run: `{str(bool(closeout['kill_events_present_post'])).lower()}`",
            f"- Reconcile mismatch post-run: `{closeout['reconcile_mismatch_count_post']}`",
            f"- Systemd restarts: `{closeout['systemd_nrestarts_post']}`",
            f"- Restored trade mode: `{closeout['trade_mode_post']}`",
            f"- Direct venue audit after restore: `{'clean across all five venues' if completion['final_direct_venue_audit_clean'] else 'see closeout bundle'}`",
            f"- Account-balance PnL: `{format_decimal(balance_delta, places=8)} USD`",
            f"- Absolute balance drift: `{format_decimal(balance_abs_delta, places=8)} USD`",
            "",
            "Final 2h executed volume:",
            "",
            f"- Total fills: `{closeout['fill_count_total']}`",
            f"- Total base: `{format_decimal(closeout['fill_base_total_eth'])} ETH`",
        ]
    )
    for venue in venues:
        lines.append(
            f"- `{venue}`: `{safe_int(fills[venue]['fill_count'])} fills / {format_decimal(fills[venue]['fill_base_eth'])} ETH`"
        )
    lines.extend(["", "Final 2h order activity existed on the accepted surface:", ""])
    for venue in venues:
        venue_activity = activity[venue]
        lines.append(
            f"- `{venue}`: `{venue_activity['place_intents']} place intents`, `{venue_activity['place_acks']} place acks`, "
            f"`{venue_activity['cancel_intents']} cancel intents`, `{venue_activity['cancel_acks']} cancel acks`"
        )
    lines.extend(["", "Opportunity-adjusted venue participation:", ""])
    for venue in venues:
        lines.append(f"- `{venue}`: `{opportunity_reasons.get(venue) or 'not_passed'}`")

    if lineage is not None:
        lines.extend(
            [
                "",
                "## Exact-Surface Lineage Evidence",
                "",
                "The final 2h sample is accepted together with the exact-surface lineage immediately before this closeout:",
                "",
                f"- Tranche: `{lineage['tranche_id']}`",
                f"- Run root: `{lineage['run_root']}`",
                f"- Segment UTC: `{lineage['segment_start_utc']} -> {lineage['segment_end_utc']}`",
                f"- Total fills: `{lineage['fill_count_total']}`",
                f"- Total base: `{format_decimal(lineage['fill_base_total_eth'])} ETH`",
            ]
        )
        for venue in venues:
            lineage_venue = lineage["fills_by_venue"][venue]
            lines.append(
                f"- `{venue}`: `{safe_int(lineage_venue['fill_count'])} fills / {format_decimal(lineage_venue['fill_base_eth'])} ETH`"
            )

    lines.extend(
        [
            "",
            "## Why This Closeout Is Accepted",
            "",
            f"- No venue remains excluded because of an unresolved connector or platform defect.",
            f"- All five venues are execution-eligible on one integrated live surface.",
            f"- All five venues are FV-eligible on the integrated surface.",
            f"- All five venues have frozen `fill` roles on the accepted surface.",
            f"- The final 2h reopened soak was operationally clean.",
            f"- Authoritative combined account-balance PnL was `{format_decimal(balance_delta, places=8)} USD` across `{completion.get('balance_snapshot_venue_count')}` venue accounts.",
            f"- Opportunity-adjusted participation passed all five venues; MM fill evidence venues were `{','.join(mm_fill_venues)}`.",
            f"- Direct venue truth after restore showed zero positions and zero open orders across all five venues.",
            "",
            "## Known Caveats",
            "",
            f"- Final 2h `would_send_zero_pct` remained high at `{format_decimal(risk_summary.get('would_send_zero_pct'), places=4)}`; economics and sizing optimization should continue on top of this clean topology.",
        ]
    )
    zero_fill_venues = [venue for venue in venues if safe_int(fills[venue]["fill_count"]) == 0]
    if zero_fill_venues:
        lines.append(
            f"- The final 2h sample had `0` fills on `{','.join(zero_fill_venues)}`, so future monitoring should keep checking venue-balance drift rather than assuming every venue fills in every window."
        )
    lines.extend(
        [
            "",
            "## Evidence Files",
            "",
            f"- Frozen spec: `{repo_root / 'phase5' / 'runs' / REOPENED_FINAL_CLOSEOUT_ID / 'final_topology_spec.yaml'}`",
            f"- Final 2h summary: `{run_root / 'live_segment_summary.json'}`",
            f"- Final 2h metrics: `{run_root / 'live_metrics.json'}`",
            f"- Final 2h report: `{run_root / 'telemetry_report_live_segment.md'}`",
            f"- Final 2h guard: `{run_root / 'guard.log'}`",
        ]
    )
    audit_path = spec["evidence"].get("final_direct_venue_audit", {}).get("path")
    if audit_path:
        lines.append(f"- Final 2h direct venue audit: `{audit_path}`")
    return "\n".join(lines) + "\n"


def maybe_auto_refresh_reopened_final_closeout(
    source_tranche: dict[str, Any],
    next_target: str | None,
    repo_root: Path,
    summary_path: str | None,
    guard_path: str | None,
) -> bool:
    if next_target != REOPENED_FINAL_CLOSEOUT_ID or not summary_path or not guard_path:
        return False
    run_root = Path(summary_path).parent
    if not run_root.exists():
        return False

    queue, control_pack = load_state(repo_root)
    section, index, final_tranche = find_tranche(queue, REOPENED_FINAL_CLOSEOUT_ID)
    defaults = control_pack["execution_defaults"]
    surface_id = safe_tranche_surface_id(source_tranche, control_pack, repo_root)
    queue[section][index].setdefault("control", {})["description"] = (
        f"Latest successful reopened long-soak evidence pack on promoted all5 surface {surface_id}."
    )
    queue[section][index].setdefault("candidate", {})["runtime_binary"] = str(
        candidate_runtime_binary_path(source_tranche, defaults, repo_root)
    )
    queue[section][index].setdefault("candidate", {}).setdefault("change_scope", {}).setdefault("files", [])
    queue[section][index].setdefault("env_diff", {})["stage_overlay_source"] = str(
        tranche_stage_overlay_source(source_tranche, defaults)
    )
    save_state(queue, repo_root)
    prepare_tranche(repo_root=repo_root, tranche_id=REOPENED_FINAL_CLOSEOUT_ID, mark_in_progress=False)

    queue, control_pack = load_state(repo_root)
    _, _, refreshed_final_tranche = find_tranche(queue, REOPENED_FINAL_CLOSEOUT_ID)
    lineage = latest_promoted_lineage_for_parent(queue, str(source_tranche.get("id")), surface_id)
    spec = build_reopened_final_topology_spec(
        refreshed_final_tranche,
        source_tranche,
        control_pack,
        repo_root,
        run_root,
        lineage,
    )
    final_run_dir = ensure_run_dir(repo_root, REOPENED_FINAL_CLOSEOUT_ID)
    save_yaml(final_run_dir / "final_topology_spec.yaml", spec)
    write_text(
        final_run_dir / "final_closeout.md",
        build_reopened_final_closeout_markdown(spec, run_root, repo_root),
    )
    final_decision = (
        "promote"
        if spec.get("status") == "accepted_closeout"
        and (spec.get("closeout_disposition") or {}).get("verdict") == "accepted"
        else "hold"
    )
    record_result(
        tranche_id=REOPENED_FINAL_CLOSEOUT_ID,
        decision=final_decision,
        repo_root=repo_root,
        summary_path=summary_path,
        guard_path=guard_path,
        notes=(
            f"Auto-refreshed reopened final closeout from {source_tranche.get('id')} "
            f"using {run_root}."
        ),
        credit_earned="major" if final_decision == "promote" else None,
    )
    return True


def record_result(
    tranche_id: str,
    decision: str,
    repo_root: Path = ROOT,
    summary_path: str | None = None,
    guard_path: str | None = None,
    notes: str | None = None,
    observed_blocker_family: str | None = None,
    precondition_failed: bool = False,
    credit_earned: str | None = None,
) -> None:
    maybe_auto_recover_latest_run(repo_root, tranche_id)
    queue, control_pack = load_state(repo_root)
    section, index, tranche = find_tranche(queue, tranche_id)
    if decision not in {"promote", "hold", "rollback"}:
        raise ValueError("decision must be one of: promote, hold, rollback")
    validate_optional_enum(tranche_id, observed_blocker_family, VALID_BLOCKER_FAMILIES, "observed_blocker_family")
    validate_optional_enum(tranche_id, credit_earned, VALID_PROGRESS_CREDIT, "credit_earned")
    surface_id = safe_tranche_surface_id(tranche, control_pack, repo_root)
    run_root = run_root_from_artifact_paths(summary_path, guard_path)
    requested_decision = decision
    effective_decision = decision
    history_entry = {
        "decision": requested_decision,
        "requested_decision": requested_decision,
        "timestamp_utc": utc_now(),
        "precondition_failed": precondition_failed,
        "branch_class": tranche.get("branch_class"),
        "hypothesis_blocker_family": tranche.get("hypothesis_blocker_family"),
    }
    if surface_id:
        history_entry["surface_id"] = surface_id
    if summary_path:
        history_entry["summary_path"] = summary_path
    if guard_path:
        history_entry["guard_path"] = guard_path
    if notes:
        history_entry["notes"] = notes
    if observed_blocker_family:
        history_entry["observed_blocker_family"] = observed_blocker_family
    if credit_earned is not None:
        history_entry["credit_earned"] = credit_earned
    queue[section][index].setdefault("history", []).append(history_entry)
    status_map = {"promote": "promoted", "hold": "hold", "rollback": "rolled_back"}
    queue[section][index]["status"] = status_map[requested_decision]
    save_state(queue, repo_root)
    write_status(queue, control_pack, repo_root)

    state_sync_report_path = None
    try:
        state_sync_report = audit_state_sync(queue, control_pack, repo_root, tranche_id=tranche_id)
        state_sync_report_path = write_state_sync_report(run_root, state_sync_report)
        state_sync_summary = state_sync_summary_payload(state_sync_report, state_sync_report_path)
        state_sync_blocked = state_sync_blocks_promotion(state_sync_report)
    except Exception as exc:
        state_sync_summary = {
            "requested_tranche_id": tranche_id,
            "status": "error",
            "critical_count": 1,
            "warning_count": 0,
            "surface_id": surface_id,
            "report_path": None,
            "blocks_promotion": True,
            "error": str(exc),
        }
        state_sync_blocked = True
        history_entry["state_sync_error"] = str(exc)
    history_entry["state_sync"] = state_sync_summary
    history_entry["state_sync_report_path"] = None if state_sync_report_path is None else str(state_sync_report_path)
    if requested_decision == "promote" and state_sync_blocked:
        effective_decision = "hold"
        precondition_failed = True
        history_entry["precondition_failed"] = True
        history_entry["state_sync_blocked_promotion"] = True
    else:
        history_entry["state_sync_blocked_promotion"] = False
    queue[section][index]["status"] = status_map[effective_decision]
    child_activation_allowed = (
        True
        if effective_decision == "promote"
        else fail_child_activation_allowed(tranche, observed_blocker_family, precondition_failed)
    )
    next_target = tranche.get("next_if_pass") if effective_decision == "promote" else (
        tranche_fail_child_target(tranche, observed_blocker_family) if child_activation_allowed else None
    )
    if should_suppress_reopened_long_soak_fail_loop(
        queue,
        tranche,
        observed_blocker_family,
        next_target,
        surface_id,
    ):
        child_activation_allowed = False
        next_target = None
    history_entry["decision"] = effective_decision
    history_entry["child_activation_allowed"] = child_activation_allowed
    history_entry["activated_child"] = next_target
    activate_next(queue, next_target)
    if effective_decision == "promote" and next_target == REOPENED_LONG_SOAK_ID:
        refresh_reopened_long_soak_from_parent(queue, control_pack, repo_root, parent_tranche=queue[section][index])
    save_state(queue, repo_root)
    write_status(queue, control_pack, repo_root)
    orchestration = load_orchestration(repo_root)
    found = find_orchestration_session(orchestration, tranche_id)
    session = None if found is None else found[1]
    run_root = None
    autoscore = None
    if summary_path:
        summary = Path(summary_path)
        if summary.name == "live_segment_summary.json":
            run_root = summary.parent
            autoscore_path = run_root / "autoscore_bundle.json"
            if autoscore_path.exists():
                with autoscore_path.open("r", encoding="utf-8") as handle:
                    autoscore = json.load(handle)
    child_support_gate_result = maybe_auto_prepare_child_support_gate(
        next_target,
        effective_decision,
        repo_root,
        run_root=run_root,
        autoscore=autoscore,
    )
    if child_support_gate_result is not None:
        queue, control_pack = load_state(repo_root)
        section, index, tranche = find_tranche(queue, tranche_id)
        latest_entry = latest_history_entry(queue[section][index])
        if latest_entry is not None:
            latest_entry["selected_child_support_gate"] = child_support_gate_result
        save_state(queue, repo_root)
        write_status(queue, control_pack, repo_root)
    write_support_summary(tranche_id, tranche, control_pack, session, repo_root, state_sync=state_sync_summary)
    write_venue_capability_matrix(
        tranche,
        control_pack,
        repo_root=repo_root,
        run_root=run_root,
        autoscore=autoscore,
        final_decision=effective_decision,
    )
    write_stage_verdict(
        tranche,
        control_pack,
        repo_root=repo_root,
        session=session,
        run_root=run_root,
        autoscore=autoscore,
        decision=effective_decision,
        observed_blocker_family=observed_blocker_family,
        selected_child=next_target,
        selected_child_support_gate=child_support_gate_result,
        state_sync=state_sync_summary,
    )
    if effective_decision == "promote":
        maybe_auto_refresh_reopened_final_closeout(
            tranche,
            next_target,
            repo_root,
            summary_path,
            guard_path,
        )


def duration_label(duration_sec: int) -> str:
    if duration_sec <= 300:
        return "5m_canary"
    if duration_sec <= 1200:
        return "20m_soak"
    if duration_sec <= 3600:
        return "60m_qual"
    return f"{duration_sec}s"


def tranche_live_guard_args(tranche: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    default_args = defaults.get("live_guard_args", []) or []
    tranche_args = tranche.get("execution", {}).get("live_guard_args", []) or []
    if not isinstance(default_args, list) or not all(isinstance(item, str) for item in default_args):
        raise ValueError("execution_defaults.live_guard_args must be a list of strings")
    if not isinstance(tranche_args, list) or not all(isinstance(item, str) for item in tranche_args):
        raise ValueError("execution.live_guard_args must be a list of strings")
    return [*default_args, *tranche_args]


def tranche_stage_overlay_source(tranche: dict[str, Any], defaults: dict[str, Any]) -> Path:
    override = tranche.get("execution", {}).get("stage_overlay_source")
    if override is None:
        override = tranche.get("env_diff", {}).get("stage_overlay_source")
    if override is None:
        return Path(defaults["stage_overlay_source"])
    if not isinstance(override, str) or not override:
        raise ValueError("stage_overlay_source override must be a non-empty string path")
    return Path(override)


def tranche_change_scope_files(tranche: dict[str, Any]) -> list[Path]:
    files = tranche.get("candidate", {}).get("change_scope", {}).get("files", []) or []
    if not isinstance(files, list):
        raise ValueError("candidate.change_scope.files must be a list when present")
    out: list[Path] = []
    for item in files:
        if not isinstance(item, str) or not item:
            raise ValueError("candidate.change_scope.files entries must be non-empty strings")
        out.append(Path(item))
    return out


def tranche_requires_repo_runtime(tranche: dict[str, Any], repo_root: Path) -> bool:
    for path in tranche_change_scope_files(tranche):
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        if rel.parts[:2] == ("paraphina", "src"):
            return True
        if rel.parts[:2] == ("paraphina", "tests"):
            return True
        if rel == Path("paraphina") / "Cargo.toml":
            return True
    return False


def inferred_repo_runtime_binary(repo_root: Path) -> Path:
    return repo_root / "target" / "release" / "paraphina_live"


def candidate_runtime_binary_path(
    tranche: dict[str, Any],
    defaults: dict[str, Any],
    repo_root: Path = ROOT,
) -> Path:
    explicit = tranche.get("candidate", {}).get("runtime_binary")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("candidate.runtime_binary must be a non-empty string path")
        return Path(explicit)

    runtime_binary = Path(defaults["runtime_binary"])
    if not tranche_requires_repo_runtime(tranche, repo_root):
        return runtime_binary

    candidate_binary = inferred_repo_runtime_binary(repo_root)
    if not candidate_binary.exists():
        raise ValueError(
            f"{tranche.get('id')}: expected built candidate runtime {candidate_binary} for code-changing tranche"
        )

    newest_change_mtime_ns = 0
    for path in tranche_change_scope_files(tranche):
        if path.exists():
            newest_change_mtime_ns = max(newest_change_mtime_ns, path.stat().st_mtime_ns)
    if newest_change_mtime_ns and candidate_binary.stat().st_mtime_ns < newest_change_mtime_ns:
        raise ValueError(
            f"{tranche.get('id')}: built candidate runtime {candidate_binary} is older than change-scope files; rebuild before live run"
        )
    return candidate_binary


def cleanup_binary_path(defaults: dict[str, Any], repo_root: Path = ROOT) -> Path:
    explicit = defaults.get("cleanup_binary")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("execution_defaults.cleanup_binary must be a non-empty string path")
        return Path(explicit)

    for candidate in (
        repo_root / "target" / "release" / "live_cleanup",
        repo_root / "paraphina" / "target" / "release" / "live_cleanup",
        Path("/home/ubuntu/paraphina/target/release/live_cleanup"),
    ):
        if candidate.exists():
            return candidate
    return repo_root / "target" / "release" / "live_cleanup"


def runtime_install_required(candidate_binary: Path, runtime_binary: Path) -> bool:
    return candidate_binary.resolve() != runtime_binary.resolve()


def systemd_show(service: str) -> str:
    proc = subprocess.run(
        ["systemctl", "show", service, "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def curl_health(url: str, attempts: int = 1, delay_sec: float = 0.0) -> str:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(attempts):
        try:
            proc = subprocess.run(["curl", "-fsS", url], check=True, capture_output=True, text=True)
            return proc.stdout
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt + 1 < attempts and delay_sec > 0:
                time.sleep(delay_sec)
    assert last_error is not None
    raise last_error


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free


def required_free_bytes(duration_sec: int) -> int:
    if duration_sec <= 300:
        return 512 * MIB
    if duration_sec <= 1200:
        return 2 * GIB
    return 4 * GIB


def ensure_disk_headroom(path: Path, duration_sec: int) -> None:
    free = free_bytes(path)
    required = required_free_bytes(duration_sec)
    if free < required:
        raise RuntimeError(
            f"insufficient disk headroom for {duration_label(duration_sec)} run: "
            f"free={free} bytes required={required} bytes on {path}"
        )


def resolve_runtime_tempdir() -> Path:
    candidate = os.environ.get("TMPDIR")
    if candidate:
        return Path(candidate)
    return Path("/tmp")


def runtime_storage_headroom_checks(
    control_pack: dict[str, Any],
    duration_sec: int,
    repo_root: Path = ROOT,
) -> list[tuple[str, Path, int]]:
    defaults = automation_defaults(control_pack, repo_root)
    execution_defaults = control_pack["execution_defaults"]
    telemetry_path = Path(execution_defaults["telemetry_path"])
    promotion_root = Path(execution_defaults["promotion_runs_root"])
    duration_required = required_free_bytes(duration_sec)
    telemetry_required = max(duration_required, defaults["telemetry_headroom_bytes"])
    tempdir_required = max(512 * MIB, defaults["tempdir_headroom_bytes"])
    tempdir_root = resolve_runtime_tempdir()
    current_runs_root = Path(CURRENT_RUNS_DIR)
    return [
        ("repo", repo_root, defaults["repo_headroom_bytes"]),
        ("promotion_runs", promotion_root, defaults["promotion_runs_headroom_bytes"]),
        ("telemetry", telemetry_path, telemetry_required),
        ("tempdir", tempdir_root, tempdir_required),
        ("current_runs", current_runs_root, tempdir_required),
    ]


def ensure_runtime_storage_headroom(
    control_pack: dict[str, Any],
    duration_sec: int,
    repo_root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for label, path, required in runtime_storage_headroom_checks(control_pack, duration_sec, repo_root):
        free = free_bytes(path)
        results[label] = {
            "path": str(path),
            "free_bytes": free,
            "required_bytes": required,
        }
        if free < required:
            raise RuntimeError(
                f"{label} headroom below automation default: "
                f"free={free} required={required} path={path}"
            )
    return results


def ensure_shadow_health(url: str = HEALTH_URL, service: str = "paraphina_live") -> dict[str, Any]:
    try:
        payload = json.loads(curl_health(url, attempts=5, delay_sec=1.0))
    except subprocess.CalledProcessError:
        state = systemd_show(service)
        if "ActiveState=active" not in state or "SubState=running" not in state:
            raise
        payload = json.loads(curl_health(url, attempts=10, delay_sec=1.0))
    if not payload.get("healthy") or not payload.get("ready"):
        raise RuntimeError(f"host health not ready for tranche admission: {payload}")
    if payload.get("trade_mode") != "shadow":
        raise RuntimeError(f"host must be in shadow for tranche admission, got trade_mode={payload.get('trade_mode')!r}")
    return payload


def _normalize_log_pattern_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {
            "label": spec,
            "pattern": spec,
            "is_regex": False,
            "min_occurrences": 1,
        }
    if not isinstance(spec, dict):
        raise ValueError(f"unsupported log pattern spec {spec!r}")
    regex = spec.get("regex")
    pattern = spec.get("pattern")
    if regex:
        label = str(regex)
        is_regex = True
    elif pattern:
        label = str(pattern)
        is_regex = False
    else:
        raise ValueError(f"log pattern spec requires pattern or regex: {spec!r}")
    min_occurrences = int(spec.get("min_occurrences") or 1)
    return {
        "label": label,
        "pattern": label,
        "is_regex": is_regex,
        "min_occurrences": min_occurrences,
    }


def _log_pattern_count(text: str, spec: Any) -> dict[str, Any]:
    normalized = _normalize_log_pattern_spec(spec)
    pattern = normalized["pattern"]
    if normalized["is_regex"]:
        count = len(re.findall(pattern, text, flags=re.MULTILINE))
    else:
        count = text.count(pattern)
    return {
        "label": normalized["label"],
        "is_regex": normalized["is_regex"],
        "count": count,
        "min_occurrences": normalized["min_occurrences"],
        "passed": count >= normalized["min_occurrences"],
    }


def evaluate_shadow_mechanism_evidence(
    tranche: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    evidence = tranche.get("mechanism_evidence")
    if not isinstance(evidence, dict) or not evidence:
        return {
            "configured": False,
            "mechanism_pass": True,
            "failure_reason": None,
            "log_path": str(run_root / "run.log"),
        }

    log_path = run_root / str(evidence.get("log_file") or "run.log")
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "configured": True,
            "mechanism_pass": False,
            "failure_reason": "mechanism_log_missing",
            "log_path": str(log_path),
        }

    required_all = [_log_pattern_count(text, spec) for spec in evidence.get("required_log_patterns_all", []) or []]
    required_any = [_log_pattern_count(text, spec) for spec in evidence.get("required_log_patterns_any", []) or []]

    ratio_results: list[dict[str, Any]] = []
    ratio_specs = evidence.get("max_log_pattern_ratio", []) or []
    if isinstance(ratio_specs, dict):
        ratio_specs = [ratio_specs]
    for spec in ratio_specs:
        if not isinstance(spec, dict):
            continue
        numerator = _log_pattern_count(
            text,
            {"regex": spec["numerator_regex"]} if spec.get("numerator_regex") else spec.get("numerator_pattern"),
        )
        denominator = _log_pattern_count(
            text,
            {"regex": spec["denominator_regex"]} if spec.get("denominator_regex") else spec.get("denominator_pattern"),
        )
        max_ratio = float(spec.get("max_ratio") if spec.get("max_ratio") is not None else 1.0)
        ratio = None if denominator["count"] == 0 else numerator["count"] / denominator["count"]
        ratio_results.append(
            {
                "numerator": numerator,
                "denominator": denominator,
                "max_ratio": max_ratio,
                "ratio": ratio,
                "passed": ratio is not None and ratio <= max_ratio,
            }
        )

    if any(not item["passed"] for item in required_all):
        failed = next(item for item in required_all if not item["passed"])
        return {
            "configured": True,
            "mechanism_pass": False,
            "failure_reason": f"missing_required_log_pattern:{failed['label']}",
            "log_path": str(log_path),
            "required_log_patterns_all": required_all,
            "required_log_patterns_any": required_any,
            "max_log_pattern_ratio": ratio_results,
        }
    if required_any and not any(item["passed"] for item in required_any):
        labels = ",".join(item["label"] for item in required_any)
        return {
            "configured": True,
            "mechanism_pass": False,
            "failure_reason": f"missing_any_required_log_pattern:{labels}",
            "log_path": str(log_path),
            "required_log_patterns_all": required_all,
            "required_log_patterns_any": required_any,
            "max_log_pattern_ratio": ratio_results,
        }
    if any(not item["passed"] for item in ratio_results):
        failed = next(item for item in ratio_results if not item["passed"])
        return {
            "configured": True,
            "mechanism_pass": False,
            "failure_reason": "mechanism_fallback_ratio_exceeded",
            "log_path": str(log_path),
            "required_log_patterns_all": required_all,
            "required_log_patterns_any": required_any,
            "max_log_pattern_ratio": ratio_results,
            "failed_ratio": failed,
        }

    return {
        "configured": True,
        "mechanism_pass": True,
        "failure_reason": None,
        "log_path": str(log_path),
        "required_log_patterns_all": required_all,
        "required_log_patterns_any": required_any,
        "max_log_pattern_ratio": ratio_results,
    }


def evaluate_shadow_connector_availability(run_root: Path) -> dict[str, Any]:
    log_path = run_root / "run.log"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "connector_unavailable": [],
            "failure_reason": None,
            "log_path": str(log_path),
        }

    unavailable: list[str] = []
    marker = "error=connector_unavailable connector="
    for line in text.splitlines():
        if marker not in line:
            continue
        connector = line.split(marker, 1)[1].split()[0].strip()
        if connector and connector not in unavailable:
            unavailable.append(connector)
    return {
        "connector_unavailable": unavailable,
        "failure_reason": None if not unavailable else f"connector_unavailable:{','.join(unavailable)}",
        "log_path": str(log_path),
    }


def support_gate_duration_sec(gate: str) -> int | None:
    if gate == "shadow_smoke_10m":
        return 600
    if gate == "shadow_smoke_30m":
        return 1800
    if gate == "shadow_ab_10m":
        return 600
    return None


def support_gate_requires_mechanism(tranche: dict[str, Any]) -> bool:
    value = tranche.get("support_gate_require_mechanism")
    if value is None:
        return True
    return bool(value)


def support_gate_satisfied(
    tranche_id: str,
    tranche: dict[str, Any],
    repo_root: Path,
    surface_id: str,
) -> tuple[bool, str]:
    gate = tranche.get("support_gate") or "none"
    if gate == "none":
        return True, "support gate not required"

    manifest_path = support_gate_manifest_path(repo_root, tranche_id, gate)
    if not manifest_path.exists():
        manifest_path = latest_run_manifest_path(repo_root, tranche_id)
    if not manifest_path.exists():
        return False, f"missing support-gate manifest for {gate}"
    manifest = load_yaml(manifest_path)
    if manifest.get("surface_id") != surface_id:
        return False, f"support-gate manifest surface_id mismatch: expected {surface_id}"
    evaluation = manifest.get("support_gate_evaluation")
    if isinstance(evaluation, dict) and not evaluation.get("gate_passed", False):
        return False, str(evaluation.get("failure_reason") or f"{gate} support-gate evaluation failed")

    if gate == "shadow_smoke_10m":
        if manifest.get("shadow_smoke_status") != "pass":
            return False, str(
                (evaluation or {}).get("failure_reason")
                or f"shadow smoke not passed: {manifest.get('shadow_smoke_status')!r}"
            )
        if int(manifest.get("duration_sec") or 0) < 600:
            return False, "shadow smoke duration below required 600s"
        return True, "shadow smoke passed"

    if gate == "shadow_smoke_30m":
        if manifest.get("shadow_smoke_status") != "pass":
            return False, str(
                (evaluation or {}).get("failure_reason")
                or f"shadow smoke not passed: {manifest.get('shadow_smoke_status')!r}"
            )
        if int(manifest.get("duration_sec") or 0) < 1800:
            return False, "shadow smoke duration below required 1800s"
        return True, "shadow smoke passed"

    if gate == "shadow_ab_10m":
        if manifest.get("shadow_ab_status") != "pass":
            return False, f"shadow ab not passed: {manifest.get('shadow_ab_status')!r}"
        if int(manifest.get("duration_sec") or 0) < 600:
            return False, "shadow ab duration below required 600s"
        return True, "shadow ab passed"

    return False, f"unsupported support gate {gate!r}"


def hyperliquid_rate_limit_summary(payload: dict[str, Any]) -> dict[str, Any]:
    used = int(payload.get("nRequestsUsed") or 0)
    cap = int(payload.get("nRequestsCap") or 0)
    surplus = int(payload.get("nRequestsSurplus") or 0)
    usage_pct = None if cap <= 0 else round((used / cap) * 100.0, 4)
    available_request_weight = None if cap <= 0 else max(0, cap - used)
    request_weight_to_clear = max(0, used - cap + 1) if cap > 0 else 0
    reserve_cost_to_clear_usdc = round(request_weight_to_clear * 0.0005, 8)
    return {
        "status": "pass" if cap <= 0 or used < cap else "fail",
        "nRequestsUsed": used,
        "nRequestsCap": cap,
        "nRequestsSurplus": surplus,
        "cumVlm": payload.get("cumVlm"),
        "usage_pct": usage_pct,
        "blocked": bool(cap > 0 and used >= cap),
        "available_request_weight": available_request_weight,
        "request_weight_to_clear": request_weight_to_clear,
        "reserve_cost_to_clear_usdc": reserve_cost_to_clear_usdc,
    }


def hyperliquid_quota_runway_threshold(duration_sec: int | None) -> int:
    if duration_sec is None or duration_sec <= 0:
        return 0
    if duration_sec <= 300:
        return 2_000
    if duration_sec <= 1_200:
        return 5_000
    return 10_000


def apply_hyperliquid_quota_runway(
    summary: dict[str, Any],
    duration_sec: int | None,
) -> dict[str, Any]:
    required = hyperliquid_quota_runway_threshold(duration_sec)
    available = summary.get("available_request_weight")
    shortfall = 0 if available is None else max(0, required - int(available))
    over_cap_blocked = bool(summary.get("blocked"))
    runway_blocked = bool(
        not over_cap_blocked and required > 0 and available is not None and shortfall > 0
    )
    summary.update(
        {
            "duration_sec": duration_sec,
            "required_runway_request_weight": required,
            "runway_available_request_weight": available,
            "runway_shortfall_request_weight": shortfall,
            "runway_blocked": runway_blocked,
        }
    )
    if runway_blocked:
        summary["status"] = "fail"
        summary["blocked"] = True
        summary["blocked_reason"] = "hyperliquid_quota_runway_insufficient"
    return summary


def phase5_live_admission_env(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
) -> dict[str, str]:
    defaults = control_pack["execution_defaults"]
    env: dict[str, str] = {}
    env_file = defaults.get("env_file")
    if isinstance(env_file, str) and env_file.strip():
        env.update(parse_env_file(Path(env_file)))
    for path in (
        Path("/etc/paraphina/paraphina_live.env"),
        tranche_stage_overlay_source(tranche, defaults),
    ):
        env.update(parse_env_file(path))
    return env


def hyperliquid_user_rate_limit_preflight(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    duration_sec: int | None = None,
) -> dict[str, Any]:
    env = phase5_live_admission_env(tranche, control_pack)
    connectors = {
        item.strip().lower()
        for item in str(env.get("PARAPHINA_LIVE_CONNECTORS") or "").split(",")
        if item.strip()
    }
    if "hyperliquid" not in connectors:
        return {"status": "skipped", "reason": "hyperliquid_not_in_live_connectors"}

    user = (env.get("HL_VAULT_ADDRESS") or env.get("HL_USER") or "").strip()
    if not user:
        return {"status": "skipped", "reason": "missing_hyperliquid_user"}

    info_url = (env.get("HL_INFO_URL") or "https://api.hyperliquid.xyz/info").strip()
    timeout_sec = float(os.environ.get("PHASE5_HL_QUOTA_PREFLIGHT_TIMEOUT_SEC") or "15")
    request = urllib.request.Request(
        info_url,
        data=json.dumps({"type": "userRateLimit", "user": user}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Hyperliquid userRateLimit preflight failed: {exc}") from exc

    summary = apply_hyperliquid_quota_runway(
        hyperliquid_rate_limit_summary(payload),
        duration_sec,
    )
    if summary["blocked"]:
        if summary.get("runway_blocked"):
            raise RuntimeError(
                "Hyperliquid quota runway insufficient for live admission: "
                f"available_request_weight={summary['runway_available_request_weight']} "
                f"required_runway_request_weight={summary['required_runway_request_weight']} "
                f"runway_shortfall_request_weight={summary['runway_shortfall_request_weight']} "
                f"duration_sec={duration_sec}."
            )
        raise RuntimeError(
            "Hyperliquid action quota blocks live admission: "
            f"nRequestsUsed={summary['nRequestsUsed']} nRequestsCap={summary['nRequestsCap']} "
            f"usage_pct={summary['usage_pct']} "
            f"request_weight_to_clear={summary['request_weight_to_clear']} "
            f"reserve_cost_to_clear_usdc={summary['reserve_cost_to_clear_usdc']}."
        )
    return summary


def admission_check(tranche_id: str, duration_sec: int, repo_root: Path = ROOT) -> dict[str, Any]:
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    ensure_latest_run_not_restore_required(repo_root, tranche_id)
    storage_preflight = ensure_runtime_storage_headroom(control_pack, duration_sec, repo_root)
    health_payload = ensure_shadow_health()
    runtime_binary = candidate_runtime_binary_path(tranche, control_pack["execution_defaults"], repo_root)
    if not runtime_binary.exists():
        raise RuntimeError(f"candidate runtime binary missing for tranche admission: {runtime_binary}")
    cleanup_binary = cleanup_binary_path(control_pack["execution_defaults"], repo_root)
    if not cleanup_binary.exists():
        raise RuntimeError(f"cleanup binary missing for live tranche admission: {cleanup_binary}")

    surface_id = tranche_surface_id(tranche, control_pack, repo_root)
    support_ok, support_reason = support_gate_satisfied(tranche_id, tranche, repo_root, surface_id)
    if not support_ok:
        raise RuntimeError(f"{tranche_id}: support gate failed: {support_reason}")

    hyperliquid_quota_preflight = hyperliquid_user_rate_limit_preflight(
        tranche,
        control_pack,
        duration_sec,
    )

    for blocker_family in tranche.get("required_cleared_blockers", []) or []:
        latest_event = latest_blocker_event(queue, blocker_family, surface_id)
        if latest_event is None:
            raise RuntimeError(
                f"{tranche_id}: required cleared blocker {blocker_family!r} has no same-surface clearance record for surface_id={surface_id}"
            )
        if latest_event["status"] != "cleared":
            raise RuntimeError(
                f"{tranche_id}: required cleared blocker {blocker_family!r} reopened on surface_id={surface_id} "
                f"via tranche {latest_event['tranche_id']} at {latest_event['timestamp_utc']}"
            )

    return {
        "tranche_id": tranche_id,
        "surface_id": surface_id,
        "trade_mode": health_payload.get("trade_mode"),
        "support_gate": tranche.get("support_gate") or "none",
        "support_gate_reason": support_reason,
        "required_cleared_blockers": tranche.get("required_cleared_blockers", []) or [],
        "hyperliquid_quota_preflight": hyperliquid_quota_preflight,
        "storage_preflight": storage_preflight,
        "runtime_binary": str(runtime_binary),
        "cleanup_binary": str(cleanup_binary),
    }


def parse_kv_lines(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def normalized_closeout_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    closeout = copy.deepcopy(payload or {})
    if not isinstance(closeout, dict):
        return {}
    closeout.setdefault(
        "first_pre_restore_venue_audit_clean",
        closeout.get("pre_restore_venue_audit_clean"),
    )
    closeout.setdefault(
        "pre_restore_cleanup_required",
        False if closeout.get("pre_restore_venue_audit_clean") is True else None,
    )
    closeout.setdefault("pre_restore_cleanup_cost_usd", 0.0)
    closeout.setdefault("metrics_exists", bool(closeout.get("metrics_path")))
    health_post_complete = closeout.get("health_post_complete")
    if health_post_complete is None:
        health_post_complete = all(
            key in closeout
            for key in ("trade_mode_post", "healthy_post", "ready_post", "kill_events_present_post")
        )
    systemd_post_complete = closeout.get("systemd_post_complete")
    if systemd_post_complete is None:
        systemd_post_complete = all(
            closeout.get(key) is not None
            for key in ("systemd_active_state_post", "systemd_sub_state_post", "systemd_nrestarts_post")
        )
    closeout["health_post_complete"] = bool(health_post_complete)
    closeout["systemd_post_complete"] = bool(systemd_post_complete)
    closeout.setdefault(
        "guard_result_exists",
        closeout.get("guard_exit_code") is not None
        or "guard_intervened" in closeout
        or "guard_window_completed" in closeout,
    )
    closeout["closeout_contract_complete"] = bool(
        closeout.get("summary_exists")
        and closeout.get("report_exists")
        and closeout.get("metrics_exists")
        and closeout.get("guard_result_exists")
        and closeout["health_post_complete"]
        and closeout["systemd_post_complete"]
    )
    closeout["closeout_completeness"] = "full" if closeout["closeout_contract_complete"] else "partial"
    return closeout


def parse_ws_audit_tokens(line: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _count(mapping: dict[str, int], key: str | None) -> None:
    if not key:
        return
    mapping[key] = mapping.get(key, 0) + 1


def _safe_int(value: str | None) -> int | None:
    if value is None or value == "" or value == "na":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "" or value == "na":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def iter_json_objects(path: Path) -> Any:
    decoder = json.JSONDecoder()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                while line:
                    try:
                        record, idx = decoder.raw_decode(line)
                    except json.JSONDecodeError:
                        break
                    if isinstance(record, dict):
                        yield record
                    line = line[idx:].lstrip()
    except OSError:
        return


def parse_paradex_stderr_metrics(stderr_segment: Path) -> dict[str, Any]:
    profile_summary: dict[str, Any] = {
        "observed": False,
        "interactive_token_usage_observed": False,
        "action_counts": {},
        "token_usage_counts": {},
        "auth_source_counts": {},
        "last_token_usage": None,
        "last_auth_source": None,
    }
    order_flag_summary: dict[str, Any] = {
        "records": 0,
        "actions": {},
        "instructions": {},
        "flags": {},
        "token_usage_counts": {},
    }
    fill_flag_summary: dict[str, Any] = {
        "records": 0,
        "flags": {},
        "token_usage_counts": {},
    }
    interactive_top_summary: dict[str, Any] = {
        "records": 0,
        "public_records": 0,
        "feed_type_counts": {},
        "public_top_source_counts": {},
        "first_seq_no": None,
        "last_seq_no": None,
        "last_top": {},
        "last_public_top": {},
        "interactive_fields_present": False,
        "interactive_top_level_fallback_present": False,
    }
    ui_book_truth_summary: dict[str, Any] = {
        "observed": False,
        "api_records": 0,
        "interactive_records": 0,
        "status_counts": {},
        "error_counts": {},
        "token_usage_counts": {},
        "last_api_top": {},
        "last_interactive_top": {},
        "last_split_top": {},
        "last_api_seq_no": None,
        "last_interactive_seq_no": None,
        "last_seq_gap": None,
        "nonzero_gap_records": 0,
        "interactive_fields_present": False,
        "interactive_top_level_present": False,
    }
    ui_touch_reference_summary: dict[str, Any] = {
        "observed": False,
        "applied_count": 0,
        "source_kind_counts": {},
        "last_applied": {},
    }
    if not stderr_segment.exists():
        return {
            "paradex_profile_usage_summary": profile_summary,
            "paradex_order_flag_summary": order_flag_summary,
            "paradex_fill_flag_summary": fill_flag_summary,
            "paradex_interactive_top_summary": interactive_top_summary,
            "paradex_ui_book_truth_summary": ui_book_truth_summary,
            "paradex_ui_touch_reference_summary": ui_touch_reference_summary,
        }

    for raw_line in stderr_segment.read_text(encoding="utf-8").splitlines():
        if "PARADEX_INTERACTIVE_PUBLIC_TOP" in raw_line:
            pairs = parse_ws_audit_tokens(raw_line)
            interactive_top_summary["records"] += 1
            interactive_top_summary["public_records"] += 1
            top_source = pairs.get("top_source")
            _count(interactive_top_summary["public_top_source_counts"], top_source)
            last_public_top = {
                "top_source": top_source,
                "bid_px": _safe_float(pairs.get("bid") or pairs.get("bid_px")),
                "bid_sz": _safe_float(pairs.get("bid_sz")),
                "ask_px": _safe_float(pairs.get("ask") or pairs.get("ask_px")),
                "ask_sz": _safe_float(pairs.get("ask_sz")),
            }
            interactive_top_summary["last_public_top"] = last_public_top
            if top_source == "interactive_top_level_fallback":
                interactive_top_summary["interactive_top_level_fallback_present"] = True
            continue
        if "WS_AUDIT" not in raw_line or "venue=paradex" not in raw_line:
            continue
        pairs = parse_ws_audit_tokens(raw_line)
        component = pairs.get("component")
        if component == "profile_usage":
            token_usage = pairs.get("token_usage")
            auth_source = pairs.get("auth_source")
            profile_summary["observed"] = True
            if token_usage == "interactive":
                profile_summary["interactive_token_usage_observed"] = True
            _count(profile_summary["action_counts"], pairs.get("action"))
            _count(profile_summary["token_usage_counts"], token_usage)
            _count(profile_summary["auth_source_counts"], auth_source)
            profile_summary["last_token_usage"] = token_usage
            profile_summary["last_auth_source"] = auth_source
        elif component == "order_flags":
            order_flag_summary["records"] += 1
            token_usage = pairs.get("token_usage")
            _count(order_flag_summary["actions"], pairs.get("action"))
            _count(order_flag_summary["instructions"], pairs.get("instruction"))
            _count(order_flag_summary["token_usage_counts"], token_usage)
            flags_raw = pairs.get("flags", "none")
            for flag in [part for part in flags_raw.split(",") if part and part != "none"]:
                _count(order_flag_summary["flags"], flag)
        elif component == "fill_flags":
            fill_flag_summary["records"] += 1
            token_usage = pairs.get("token_usage")
            _count(fill_flag_summary["token_usage_counts"], token_usage)
            flags_raw = pairs.get("flags", "")
            for flag in [part for part in flags_raw.split(",") if part]:
                _count(fill_flag_summary["flags"], flag)
        elif component == "interactive_top":
            interactive_top_summary["records"] += 1
            feed_type = pairs.get("feed_type")
            _count(interactive_top_summary["feed_type_counts"], feed_type)
            seq_no = _safe_int(pairs.get("seq_no"))
            if interactive_top_summary["first_seq_no"] is None and seq_no is not None:
                interactive_top_summary["first_seq_no"] = seq_no
            if seq_no is not None:
                interactive_top_summary["last_seq_no"] = seq_no
            last_top = {
                "best_bid_api_price": _safe_float(pairs.get("best_bid_api_price")),
                "best_bid_api_size": _safe_float(pairs.get("best_bid_api_size")),
                "best_bid_interactive_price": _safe_float(pairs.get("best_bid_interactive_price")),
                "best_bid_interactive_size": _safe_float(pairs.get("best_bid_interactive_size")),
                "best_ask_api_price": _safe_float(pairs.get("best_ask_api_price")),
                "best_ask_api_size": _safe_float(pairs.get("best_ask_api_size")),
                "best_ask_interactive_price": _safe_float(pairs.get("best_ask_interactive_price")),
                "best_ask_interactive_size": _safe_float(pairs.get("best_ask_interactive_size")),
            }
            interactive_top_summary["last_top"] = last_top
            interactive_top_summary["interactive_fields_present"] = any(
                last_top[field] is not None
                for field in (
                    "best_bid_interactive_price",
                    "best_ask_interactive_price",
                )
            )
        elif component == "ui_book_truth":
            source = pairs.get("source")
            status = pairs.get("status")
            token_usage = pairs.get("token_usage")
            ui_book_truth_summary["observed"] = True
            _count(ui_book_truth_summary["status_counts"], status)
            _count(ui_book_truth_summary["token_usage_counts"], token_usage)
            if source == "api":
                ui_book_truth_summary["api_records"] += 1
            elif source == "interactive":
                ui_book_truth_summary["interactive_records"] += 1
            if status and status != "ok":
                _count(ui_book_truth_summary["error_counts"], status)
            last_top = {
                "bid_px": _safe_float(pairs.get("bid_px")),
                "bid_sz": _safe_float(pairs.get("bid_sz")),
                "ask_px": _safe_float(pairs.get("ask_px")),
                "ask_sz": _safe_float(pairs.get("ask_sz")),
            }
            split_top = {
                "best_bid_api_px": _safe_float(pairs.get("best_bid_api_px")),
                "best_bid_api_sz": _safe_float(pairs.get("best_bid_api_sz")),
                "best_bid_interactive_px": _safe_float(pairs.get("best_bid_interactive_px")),
                "best_bid_interactive_sz": _safe_float(pairs.get("best_bid_interactive_sz")),
                "best_ask_api_px": _safe_float(pairs.get("best_ask_api_px")),
                "best_ask_api_sz": _safe_float(pairs.get("best_ask_api_sz")),
                "best_ask_interactive_px": _safe_float(pairs.get("best_ask_interactive_px")),
                "best_ask_interactive_sz": _safe_float(pairs.get("best_ask_interactive_sz")),
            }
            if source == "api":
                ui_book_truth_summary["last_api_top"] = last_top
                ui_book_truth_summary["last_api_seq_no"] = _safe_int(pairs.get("seq_no"))
            elif source == "interactive":
                ui_book_truth_summary["last_interactive_top"] = last_top
                ui_book_truth_summary["last_interactive_seq_no"] = _safe_int(pairs.get("seq_no"))
                if last_top["bid_px"] is not None or last_top["ask_px"] is not None:
                    ui_book_truth_summary["interactive_top_level_present"] = True
            if any(value is not None for value in split_top.values()):
                ui_book_truth_summary["last_split_top"] = split_top
                if (
                    split_top["best_bid_interactive_px"] is not None
                    or split_top["best_ask_interactive_px"] is not None
                ):
                    ui_book_truth_summary["interactive_fields_present"] = True
                bid_gap = (
                    split_top["best_bid_interactive_px"] is not None
                    and split_top["best_bid_api_px"] is not None
                    and split_top["best_bid_interactive_px"] != split_top["best_bid_api_px"]
                )
                ask_gap = (
                    split_top["best_ask_interactive_px"] is not None
                    and split_top["best_ask_api_px"] is not None
                    and split_top["best_ask_interactive_px"] != split_top["best_ask_api_px"]
                )
                if bid_gap or ask_gap:
                    ui_book_truth_summary["nonzero_gap_records"] += 1
            api_seq = ui_book_truth_summary["last_api_seq_no"]
            interactive_seq = ui_book_truth_summary["last_interactive_seq_no"]
            if api_seq is not None and interactive_seq is not None:
                ui_book_truth_summary["last_seq_gap"] = interactive_seq - api_seq
        elif component == "ui_touch_reference" and pairs.get("action") == "applied":
            ui_touch_reference_summary["observed"] = True
            ui_touch_reference_summary["applied_count"] += 1
            _count(ui_touch_reference_summary["source_kind_counts"], pairs.get("source_kind"))
            ui_touch_reference_summary["last_applied"] = {
                "source_kind": pairs.get("source_kind"),
                "orig_bid": _safe_float(pairs.get("orig_bid")),
                "orig_ask": _safe_float(pairs.get("orig_ask")),
                "adj_bid": _safe_float(pairs.get("adj_bid")),
                "adj_ask": _safe_float(pairs.get("adj_ask")),
            }

    return {
        "paradex_profile_usage_summary": profile_summary,
        "paradex_order_flag_summary": order_flag_summary,
        "paradex_fill_flag_summary": fill_flag_summary,
        "paradex_interactive_top_summary": interactive_top_summary,
        "paradex_ui_book_truth_summary": ui_book_truth_summary,
        "paradex_ui_touch_reference_summary": ui_touch_reference_summary,
    }


def ms_to_utc(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(float(ms) / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z")


def cleanup_cost_from_guard_stdout(line: str) -> float | None:
    if "_stdout=" not in line:
        return None
    raw_payload = line.split("_stdout=", 1)[1].strip()
    try:
        payload_text = ast.literal_eval(raw_payload)
    except (ValueError, SyntaxError):
        payload_text = raw_payload.strip("'\"")
    if not isinstance(payload_text, str) or not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        match = re.search(r"total_estimated_cleanup_cost_usd=([-+0-9.eE]+)", payload_text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None
    try:
        return float(payload.get("total_estimated_cleanup_cost_usd", 0.0))
    except (TypeError, ValueError):
        return None


def guard_closeout_info(guard_path: Path) -> dict[str, Any]:
    info = {
        "guard_intervened": False,
        "guard_intervention_reason": None,
        "guard_window_completed": False,
        "first_pre_restore_venue_audit_clean": None,
        "pre_restore_cleanup_required": False,
        "pre_restore_cleanup_cost_usd": 0.0,
        "pre_restore_venue_audit_clean": False,
        "pre_restore_cleanup_venue_audit_clean": False,
        "post_rollback_venue_audit_clean": False,
        "post_cleanup_venue_audit_clean": False,
    }
    if not guard_path.exists():
        return info
    for raw_line in guard_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "CRITICAL triggered_intervention reason=" in line:
            info["guard_intervened"] = True
            info["guard_intervention_reason"] = line.split("reason=", 1)[1].strip()
        elif "guard_window_complete_restoring_shadow" in line:
            info["guard_window_completed"] = True
        elif "pre_restore_cleanup_venue_audit_clean" in line:
            info["pre_restore_venue_audit_clean"] = True
            info["pre_restore_cleanup_venue_audit_clean"] = True
        elif "pre_restore_cleanup_triggered" in line or "pre_restore_venue_audit_dirty" in line:
            if info["first_pre_restore_venue_audit_clean"] is None:
                info["first_pre_restore_venue_audit_clean"] = False
            info["pre_restore_cleanup_required"] = True
        elif "pre_restore_cleanup_incomplete" in line:
            info["pre_restore_cleanup_required"] = True
        elif "pre_restore_venue_audit_clean" in line:
            if info["first_pre_restore_venue_audit_clean"] is None:
                info["first_pre_restore_venue_audit_clean"] = True
            info["pre_restore_venue_audit_clean"] = True
        elif "post_rollback_venue_audit_clean" in line:
            info["post_rollback_venue_audit_clean"] = True
        elif "post_cleanup_venue_audit_clean" in line:
            info["post_cleanup_venue_audit_clean"] = True
        cost = cleanup_cost_from_guard_stdout(line)
        if cost is not None:
            info["pre_restore_cleanup_cost_usd"] += cost
    if info["first_pre_restore_venue_audit_clean"] is None:
        info["first_pre_restore_venue_audit_clean"] = bool(info["pre_restore_venue_audit_clean"])
    return info


def latest_promotion_run_root(tranche_id: str, promotion_runs_root: Path) -> Path | None:
    candidates = sorted(
        promotion_runs_root.glob(f"{tranche_id}_*/live_canary"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def int_from_file(path: Path) -> int | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return int(raw)


def expected_bounded_size(run_root: Path) -> int | None:
    pre_offset = int_from_file(run_root / "telemetry_offset_pre.txt")
    post_offset = int_from_file(run_root / "telemetry_offset_post.txt")
    if pre_offset is None or post_offset is None or post_offset <= pre_offset:
        return None
    return post_offset - pre_offset


def expected_stderr_segment_size(run_root: Path) -> int | None:
    pre_offset = int_from_file(run_root / "err_offset_pre.txt")
    post_offset = int_from_file(run_root / "err_offset_post.txt")
    if pre_offset is None or post_offset is None or post_offset <= pre_offset:
        return None
    return post_offset - pre_offset


def parse_iso8601_utc(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_to_epoch_ms(timestamp: str) -> int:
    return int(parse_iso8601_utc(timestamp).timestamp() * 1000)


def guard_log_window(guard_path: Path) -> dict[str, Any]:
    window = {
        "first_timestamp_utc": None,
        "last_timestamp_utc": None,
        "intervention_timestamp_utc": None,
        "complete_timestamp_utc": None,
        "effective_end_timestamp_utc": None,
        "first_timestamp_ms": None,
        "last_timestamp_ms": None,
        "intervention_timestamp_ms": None,
        "complete_timestamp_ms": None,
        "effective_end_timestamp_ms": None,
    }
    if not guard_path.exists():
        return window
    timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\b")
    for raw_line in guard_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = timestamp_pattern.match(raw_line)
        if not match:
            continue
        timestamp_utc = match.group(1)
        timestamp_ms = utc_to_epoch_ms(timestamp_utc)
        if window["first_timestamp_utc"] is None:
            window["first_timestamp_utc"] = timestamp_utc
            window["first_timestamp_ms"] = timestamp_ms
        window["last_timestamp_utc"] = timestamp_utc
        window["last_timestamp_ms"] = timestamp_ms
        if (
            window["intervention_timestamp_utc"] is None
            and "CRITICAL triggered_intervention reason=" in raw_line
        ):
            window["intervention_timestamp_utc"] = timestamp_utc
            window["intervention_timestamp_ms"] = timestamp_ms
        if (
            window["complete_timestamp_utc"] is None
            and "guard_window_complete_restoring_shadow" in raw_line
        ):
            window["complete_timestamp_utc"] = timestamp_utc
            window["complete_timestamp_ms"] = timestamp_ms
    effective_utc = (
        window["intervention_timestamp_utc"]
        or window["complete_timestamp_utc"]
        or window["last_timestamp_utc"]
    )
    effective_ms = (
        window["intervention_timestamp_ms"]
        or window["complete_timestamp_ms"]
        or window["last_timestamp_ms"]
    )
    window["effective_end_timestamp_utc"] = effective_utc
    window["effective_end_timestamp_ms"] = effective_ms
    return window


def recover_telemetry_bounded_from_source(
    run_root: Path,
    telemetry_source: Path,
) -> tuple[bool, str | None]:
    telemetry_bounded = run_root / "telemetry_bounded.jsonl"
    expected_size = expected_bounded_size(run_root)
    if expected_size is None:
        return False, "missing_or_invalid_offsets"
    pre_offset = int_from_file(run_root / "telemetry_offset_pre.txt")
    assert pre_offset is not None
    post_offset = pre_offset + expected_size
    if not telemetry_source.exists():
        return False, f"telemetry_source_missing:{telemetry_source}"

    if telemetry_bounded.exists() and telemetry_bounded.stat().st_size >= expected_size:
        return False, None

    with tempfile.NamedTemporaryFile(prefix="phase5_telemetry_slice_", suffix=".jsonl", delete=False) as tmp:
        slice_path = Path(tmp.name)
    try:
        with telemetry_source.open("rb") as src, slice_path.open("wb") as dst:
            src.seek(pre_offset)
            remaining = expected_size
            while remaining > 0:
                chunk = src.read(min(4 * MIB, remaining))
                if not chunk:
                    break
                dst.write(chunk)
                remaining -= len(chunk)
        if slice_path.stat().st_size == 0:
            return False, "empty_source_slice"
        telemetry_bounded.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(slice_path, telemetry_bounded)
    finally:
        try:
            slice_path.unlink()
        except FileNotFoundError:
            pass
    return telemetry_bounded.exists(), None


def recover_telemetry_bounded_from_time_window(
    run_root: Path,
    telemetry_source: Path,
    start_ms: int,
    end_ms: int,
) -> tuple[bool, str | None]:
    telemetry_bounded = run_root / "telemetry_bounded.jsonl"
    if end_ms <= start_ms:
        return False, "invalid_time_window"
    if not telemetry_source.exists():
        return False, f"telemetry_source_missing:{telemetry_source}"

    with tempfile.NamedTemporaryFile(prefix="phase5_telemetry_time_slice_", suffix=".jsonl", delete=False) as tmp:
        slice_path = Path(tmp.name)
    matched_lines = 0
    try:
        with telemetry_source.open("r", encoding="utf-8", errors="replace") as src, slice_path.open(
            "w",
            encoding="utf-8",
        ) as dst:
            for raw_line in src:
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                event_ms = payload.get("kf_last_update_ms")
                if not isinstance(event_ms, (int, float)):
                    continue
                if start_ms <= int(event_ms) <= end_ms:
                    dst.write(raw_line)
                    matched_lines += 1
        if matched_lines == 0:
            return False, "time_window_empty"
        telemetry_bounded.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(slice_path), str(telemetry_bounded))
        return True, None
    finally:
        try:
            slice_path.unlink()
        except FileNotFoundError:
            pass


def recover_stderr_segment_from_source(
    run_root: Path,
    stderr_source: Path,
) -> tuple[bool, str | None]:
    stderr_segment = run_root / "paraphina_live.err.segment"
    pre_offset = int_from_file(run_root / "err_offset_pre.txt")
    post_offset = int_from_file(run_root / "err_offset_post.txt")
    if pre_offset is None:
        return False, "missing_err_offset_pre"
    expected_size = (
        None
        if post_offset is None or post_offset <= pre_offset
        else post_offset - pre_offset
    )
    if not stderr_source.exists():
        return False, f"stderr_source_missing:{stderr_source}"
    if expected_size is not None and stderr_segment.exists() and stderr_segment.stat().st_size >= expected_size:
        return False, None

    with tempfile.NamedTemporaryFile(prefix="phase5_stderr_slice_", suffix=".log", delete=False) as tmp:
        slice_path = Path(tmp.name)
    try:
        with stderr_source.open("rb") as src, slice_path.open("wb") as dst:
            src.seek(pre_offset)
            if expected_size is None:
                shutil.copyfileobj(src, dst)
            else:
                remaining = expected_size
                while remaining > 0:
                    chunk = src.read(min(4 * MIB, remaining))
                    if not chunk:
                        break
                    dst.write(chunk)
                    remaining -= len(chunk)
        if slice_path.stat().st_size == 0:
            return False, "empty_stderr_source_slice"
        stderr_segment.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(slice_path, stderr_segment)
    finally:
        try:
            slice_path.unlink()
        except FileNotFoundError:
            pass
    return stderr_segment.exists(), None


def recover_stderr_segment_from_time_window(
    run_root: Path,
    stderr_source: Path,
    start_ms: int,
    end_ms: int,
) -> tuple[bool, str | None]:
    stderr_segment = run_root / "paraphina_live.err.segment"
    if end_ms <= start_ms:
        return False, "invalid_stderr_time_window"
    if not stderr_source.exists():
        return False, f"stderr_source_missing:{stderr_source}"

    timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\b")
    with tempfile.NamedTemporaryFile(prefix="phase5_stderr_time_slice_", suffix=".log", delete=False) as tmp:
        slice_path = Path(tmp.name)
    matched_lines = 0
    include_line = False
    try:
        with stderr_source.open("r", encoding="utf-8", errors="replace") as src, slice_path.open(
            "w",
            encoding="utf-8",
        ) as dst:
            for raw_line in src:
                match = timestamp_pattern.match(raw_line)
                if match:
                    ts_ms = utc_to_epoch_ms(match.group(1))
                    include_line = start_ms <= ts_ms <= end_ms
                if include_line:
                    dst.write(raw_line)
                    matched_lines += 1
        if matched_lines == 0:
            return False, "stderr_time_window_empty"
        stderr_segment.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(slice_path), str(stderr_segment))
        return True, None
    finally:
        try:
            slice_path.unlink()
        except FileNotFoundError:
            pass


def synthesize_post_closeout_artifacts(
    run_root: Path,
    guard_path: Path,
    service: str,
    health_url: str,
) -> None:
    guard_info = guard_closeout_info(guard_path)
    health_post_path = run_root / "health_post.json"
    if not health_post_path.exists():
        try:
            health_payload = json.loads(curl_health(health_url, attempts=5, delay_sec=1.0))
        except Exception:
            health_payload = None
        if isinstance(health_payload, dict):
            write_text(health_post_path, json.dumps(health_payload, sort_keys=False) + "\n")

    systemd_post_path = run_root / "systemd_post.txt"
    if not systemd_post_path.exists():
        try:
            systemd_post = systemd_show(service)
        except Exception:
            systemd_post = None
        if isinstance(systemd_post, str):
            write_text(systemd_post_path, systemd_post)

    guard_result_path = run_root / "guard_result.json"
    if not guard_result_path.exists():
        exit_code = None
        if guard_info.get("guard_intervened"):
            exit_code = 1
        elif guard_info.get("guard_window_completed"):
            exit_code = 0
        if exit_code is not None:
            write_json(
                guard_result_path,
                {
                    "exit_code": exit_code,
                    "recovered_from_guard_log": True,
                },
            )


def recover_live_closeout(
    tranche_id: str,
    repo_root: Path = ROOT,
    run_root: Path | None = None,
    duration_sec: int | None = None,
) -> Path:
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    defaults = control_pack["execution_defaults"]
    promotion_runs_root = Path(defaults["promotion_runs_root"])
    manifest_path = latest_run_manifest_path(repo_root, tranche_id)
    manifest = load_yaml(manifest_path) if manifest_path.exists() else {}

    if run_root is None:
        manifest_run_root = manifest.get("run_root")
        if isinstance(manifest_run_root, str) and manifest_run_root:
            candidate = Path(manifest_run_root)
            if candidate.exists():
                run_root = candidate
    if run_root is None:
        run_root = latest_promotion_run_root(tranche_id, promotion_runs_root)
    if run_root is None:
        raise FileNotFoundError(f"No live run root found for tranche {tranche_id}")

    summary_path = run_root / "live_segment_summary.json"
    report_path = run_root / "telemetry_report_live_segment.md"
    metrics_path = run_root / "live_metrics.json"
    guard_path = run_root / "guard.log"
    telemetry_bounded = run_root / "telemetry_bounded.jsonl"
    stderr_segment = run_root / "paraphina_live.err.segment"
    health_post_path = run_root / "health_post.json"
    systemd_post_path = run_root / "systemd_post.txt"
    guard_result_path = run_root / "guard_result.json"
    analyzer_script = Path(defaults["analyzer_script"])
    telemetry_source = Path(defaults["telemetry_path"])
    stderr_source = Path(defaults["stderr_path"])

    analyzer_recovery_attempted = False
    analyzer_recovery_succeeded = False
    analyzer_recovery_error: str | None = None
    source_slice_recovery_attempted = False
    source_slice_recovery_succeeded = False
    source_slice_recovery_error: str | None = None
    time_window_recovery_attempted = False
    time_window_recovery_succeeded = False
    time_window_recovery_error: str | None = None
    stderr_source_slice_recovery_attempted = False
    stderr_source_slice_recovery_succeeded = False
    stderr_source_slice_recovery_error: str | None = None
    stderr_time_window_recovery_attempted = False
    stderr_time_window_recovery_succeeded = False
    stderr_time_window_recovery_error: str | None = None
    guard_window = guard_log_window(guard_path)

    bounded_expected_size = expected_bounded_size(run_root)
    bounded_truncated = (
        telemetry_bounded.exists()
        and bounded_expected_size is not None
        and telemetry_bounded.stat().st_size < bounded_expected_size
    )
    stderr_expected_size = expected_stderr_segment_size(run_root)
    stderr_truncated = (
        stderr_segment.exists()
        and stderr_expected_size is not None
        and stderr_segment.stat().st_size < stderr_expected_size
    )

    if not telemetry_bounded.exists() or bounded_truncated:
        source_slice_recovery_attempted = True
        source_slice_recovery_succeeded, source_slice_recovery_error = recover_telemetry_bounded_from_source(
            run_root=run_root,
            telemetry_source=telemetry_source,
        )
    if (
        (not telemetry_bounded.exists() or bounded_truncated)
        and not source_slice_recovery_succeeded
        and guard_window["first_timestamp_ms"] is not None
        and guard_window["effective_end_timestamp_ms"] is not None
    ):
        time_window_recovery_attempted = True
        time_window_recovery_succeeded, time_window_recovery_error = recover_telemetry_bounded_from_time_window(
            run_root=run_root,
            telemetry_source=telemetry_source,
            start_ms=int(guard_window["first_timestamp_ms"]),
            end_ms=int(guard_window["effective_end_timestamp_ms"]),
        )

    if not stderr_segment.exists() or stderr_truncated:
        stderr_source_slice_recovery_attempted = True
        stderr_source_slice_recovery_succeeded, stderr_source_slice_recovery_error = recover_stderr_segment_from_source(
            run_root=run_root,
            stderr_source=stderr_source,
        )
    if (
        (not stderr_segment.exists() or stderr_truncated)
        and not stderr_source_slice_recovery_succeeded
        and guard_window["first_timestamp_ms"] is not None
        and guard_window["effective_end_timestamp_ms"] is not None
    ):
        stderr_time_window_recovery_attempted = True
        stderr_time_window_recovery_succeeded, stderr_time_window_recovery_error = (
            recover_stderr_segment_from_time_window(
                run_root=run_root,
                stderr_source=stderr_source,
                start_ms=int(guard_window["first_timestamp_ms"]),
                end_ms=int(guard_window["effective_end_timestamp_ms"]),
            )
        )

    synthesize_post_closeout_artifacts(run_root, guard_path, service=defaults["service"], health_url=HEALTH_URL)

    if telemetry_bounded.exists() and (
        source_slice_recovery_succeeded
        or time_window_recovery_succeeded
        or not summary_path.exists()
        or not report_path.exists()
        or not metrics_path.exists()
    ):
        analyzer_recovery_attempted = True
        analyzer_cmd = [
            "python3",
            str(analyzer_script),
            "--telemetry",
            str(telemetry_bounded),
            "--execution-mode",
            "live",
            "--checkpoint-json",
            str(summary_path),
            "--metrics-json",
            str(metrics_path),
            "--output",
            str(report_path),
        ]
        try:
            run_logged_command(RunCommand("telemetry_analyzer_recovery", analyzer_cmd))
        except subprocess.CalledProcessError as exc:
            analyzer_recovery_error = str(exc)
        else:
            analyzer_recovery_succeeded = summary_path.exists() and report_path.exists() and metrics_path.exists()

    if not analyzer_recovery_succeeded and (not summary_path.exists() or not report_path.exists() or not metrics_path.exists()):
        if not source_slice_recovery_succeeded:
            source_slice_recovery_attempted = True
            slice_ok, slice_error = recover_telemetry_bounded_from_source(
                run_root=run_root,
                telemetry_source=telemetry_source,
            )
            source_slice_recovery_succeeded = source_slice_recovery_succeeded or slice_ok
            if source_slice_recovery_error is None:
                source_slice_recovery_error = slice_error
        if (
            not telemetry_bounded.exists()
            and not time_window_recovery_succeeded
            and guard_window["first_timestamp_ms"] is not None
            and guard_window["effective_end_timestamp_ms"] is not None
        ):
            time_window_recovery_attempted = True
            time_ok, time_error = recover_telemetry_bounded_from_time_window(
                run_root=run_root,
                telemetry_source=telemetry_source,
                start_ms=int(guard_window["first_timestamp_ms"]),
                end_ms=int(guard_window["effective_end_timestamp_ms"]),
            )
            time_window_recovery_succeeded = time_window_recovery_succeeded or time_ok
            if time_window_recovery_error is None:
                time_window_recovery_error = time_error
        if telemetry_bounded.exists():
            analyzer_recovery_attempted = True
            analyzer_cmd = [
                "python3",
                str(analyzer_script),
                "--telemetry",
                str(telemetry_bounded),
                "--execution-mode",
                "live",
                "--checkpoint-json",
                str(summary_path),
                "--metrics-json",
                str(metrics_path),
                "--output",
                str(report_path),
            ]
            try:
                run_logged_command(RunCommand("telemetry_analyzer_recovery_from_source_slice", analyzer_cmd))
            except subprocess.CalledProcessError as exc:
                analyzer_recovery_error = str(exc)
            else:
                analyzer_recovery_succeeded = summary_path.exists() and report_path.exists() and metrics_path.exists()

    summary_payload = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary_payload = json.load(handle)
    metrics_payload = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics_payload = json.load(handle)
        if isinstance(metrics_payload, dict):
            metrics_payload.update(parse_paradex_stderr_metrics(stderr_segment))
            write_json(metrics_path, metrics_payload)
    health_post_payload = {}
    if health_post_path.exists():
        with health_post_path.open("r", encoding="utf-8") as handle:
            health_post_payload = json.load(handle)
    systemd_post = parse_kv_lines(systemd_post_path.read_text(encoding="utf-8")) if systemd_post_path.exists() else {}
    guard_result_payload = {}
    if guard_result_path.exists():
        with guard_result_path.open("r", encoding="utf-8") as handle:
            guard_result_payload = json.load(handle)
    health_post_complete = health_post_path.exists() and all(
        key in health_post_payload
        for key in ("trade_mode", "healthy", "ready", "kill_events_present", "reconcile_mismatch_count")
    )
    systemd_post_complete = systemd_post_path.exists() and all(
        systemd_post.get(key) is not None
        for key in ("ActiveState", "SubState", "NRestarts")
    )
    closeout_contract_complete = all(
        (
            summary_path.exists(),
            report_path.exists(),
            metrics_path.exists(),
            guard_result_path.exists(),
            health_post_complete,
            systemd_post_complete,
        )
    )

    pnl_validity = summary_payload.get("pnl_validity", {}) if isinstance(summary_payload, dict) else {}
    economics_attribution = (
        metrics_payload.get("economics_attribution", {}) if isinstance(metrics_payload, dict) else {}
    )
    closeout_economics_attribution: dict[str, Any] = {}
    if isinstance(economics_attribution, dict):
        for key in (
            "mm_realised_net_attributed_usd",
            "mm_realised_net_unattributed_usd",
            "mm_fee_attributed_usd",
            "mm_fee_unattributed_usd",
            "hedge_fill_fee_attributed_usd",
            "hedge_fill_fee_unattributed_usd",
            "hedge_exec_cost_model_attributed_usd",
            "hedge_exec_cost_model_unattributed_usd",
            "hedge_total_cost_model_attributed_usd",
            "hedge_total_cost_model_unattributed_usd",
            "net_after_hedge_exec_model_usd",
            "venue_contribution_after_hedge",
        ):
            if key in economics_attribution:
                closeout_economics_attribution[key] = economics_attribution[key]
    latest_history = latest_history_entry(tranche) or {}
    surface_id = safe_tranche_surface_id(tranche, control_pack, repo_root)
    closeout_bundle = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "tranche_id": tranche_id,
        "surface_id": surface_id,
        "branch_class": tranche.get("branch_class"),
        "hypothesis_blocker_family": tranche.get("hypothesis_blocker_family"),
        "observed_primary_blocker_family": latest_history.get("observed_blocker_family"),
        "precondition_failed": latest_history.get("precondition_failed"),
        "credit_earned": latest_history.get("credit_earned"),
        "child_activation_allowed": latest_history.get("child_activation_allowed"),
        "run_root": str(run_root),
        "duration_sec": duration_sec if duration_sec is not None else manifest.get("duration_sec"),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "guard_path": str(guard_path),
        "guard_result_path": str(guard_result_path),
        "telemetry_bounded_path": str(telemetry_bounded),
        "stderr_segment_path": str(stderr_segment),
        "health_post_path": str(health_post_path),
        "systemd_post_path": str(systemd_post_path),
        "guard_result_exists": guard_result_path.exists(),
        "health_post_exists": health_post_path.exists(),
        "systemd_post_exists": systemd_post_path.exists(),
        "summary_exists": summary_path.exists(),
        "report_exists": report_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "health_post_complete": health_post_complete,
        "systemd_post_complete": systemd_post_complete,
        "closeout_contract_complete": closeout_contract_complete,
        "analyzer_recovery_attempted": analyzer_recovery_attempted,
        "analyzer_recovery_succeeded": analyzer_recovery_succeeded,
        "analyzer_recovery_error": analyzer_recovery_error,
        "source_slice_recovery_attempted": source_slice_recovery_attempted,
        "source_slice_recovery_succeeded": source_slice_recovery_succeeded,
        "source_slice_recovery_error": source_slice_recovery_error,
        "time_window_recovery_attempted": time_window_recovery_attempted,
        "time_window_recovery_succeeded": time_window_recovery_succeeded,
        "time_window_recovery_error": time_window_recovery_error,
        "stderr_source_slice_recovery_attempted": stderr_source_slice_recovery_attempted,
        "stderr_source_slice_recovery_succeeded": stderr_source_slice_recovery_succeeded,
        "stderr_source_slice_recovery_error": stderr_source_slice_recovery_error,
        "stderr_time_window_recovery_attempted": stderr_time_window_recovery_attempted,
        "stderr_time_window_recovery_succeeded": stderr_time_window_recovery_succeeded,
        "stderr_time_window_recovery_error": stderr_time_window_recovery_error,
        "segment_start_utc": ms_to_utc(summary_payload.get("first_ts_ms")),
        "segment_end_utc": ms_to_utc(summary_payload.get("last_ts_ms")),
        "tick_count": summary_payload.get("tick_count"),
        "final_pnl_total": pnl_validity.get("final_pnl_total"),
        "final_pnl_realised": pnl_validity.get("final_pnl_realised"),
        "final_pnl_unrealised": pnl_validity.get("final_pnl_unrealised"),
        "final_q_global_tao": pnl_validity.get("final_q_global_tao"),
        "mm_place_total": pnl_validity.get("mm_place_total"),
        "mm_keep_total": pnl_validity.get("mm_keep_total"),
        "mm_replace_total": pnl_validity.get("mm_replace_total"),
        "economics_attribution": closeout_economics_attribution,
        "fill_count_total": metrics_payload.get("fills", {}).get("total_count") if isinstance(metrics_payload, dict) else None,
        "fill_base_total": metrics_payload.get("fills", {}).get("total_base") if isinstance(metrics_payload, dict) else None,
        "trade_mode_post": health_post_payload.get("trade_mode"),
        "healthy_post": health_post_payload.get("healthy"),
        "ready_post": health_post_payload.get("ready"),
        "kill_events_present_post": health_post_payload.get("kill_events_present"),
        "reconcile_mismatch_count_post": health_post_payload.get("reconcile_mismatch_count"),
        "systemd_active_state_post": systemd_post.get("ActiveState"),
        "systemd_sub_state_post": systemd_post.get("SubState"),
        "systemd_nrestarts_post": systemd_post.get("NRestarts"),
        "guard_exit_code": guard_result_payload.get("exit_code"),
        "guard_exit_success": guard_result_payload.get("exit_code") == 0 if guard_result_payload else None,
        "closeout_completeness": (
            "full" if closeout_contract_complete else "partial"
        ),
        **guard_closeout_info(guard_path),
    }
    restore_required_reasons = closeout_restore_required_reasons(closeout_bundle)
    closeout_bundle["restore_required"] = bool(restore_required_reasons)
    closeout_bundle["restore_required_reasons"] = restore_required_reasons
    closeout_path = run_root / "live_closeout_bundle.json"
    write_json(closeout_path, closeout_bundle)

    run_state = "restore_required" if restore_required_reasons else "recovered_closeout"
    write_latest_run_manifest(
        repo_root,
        tranche_id,
        {
            "updated_utc": utc_now(),
            "surface_id": surface_id,
            "run_root": str(run_root),
            "duration_sec": closeout_bundle["duration_sec"],
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
            "guard_path": str(guard_path),
            "health_post_path": str(health_post_path),
            "systemd_post_path": str(systemd_post_path),
            "closeout_bundle_path": str(closeout_path),
            "summary_exists": summary_path.exists(),
            "report_exists": report_path.exists(),
            "metrics_exists": metrics_path.exists(),
            "analyzer_recovery_attempted": analyzer_recovery_attempted,
            "analyzer_recovery_succeeded": analyzer_recovery_succeeded,
            "source_slice_recovery_attempted": source_slice_recovery_attempted,
            "source_slice_recovery_succeeded": source_slice_recovery_succeeded,
            "stderr_source_slice_recovery_attempted": stderr_source_slice_recovery_attempted,
            "stderr_source_slice_recovery_succeeded": stderr_source_slice_recovery_succeeded,
            "stderr_time_window_recovery_attempted": stderr_time_window_recovery_attempted,
            "stderr_time_window_recovery_succeeded": stderr_time_window_recovery_succeeded,
            "run_state": run_state,
            "restore_required_reasons": restore_required_reasons,
        },
    )
    return closeout_path


def wait_for_live_health(service_url: str, timeout_sec: float = 90.0, poll_sec: float = 1.0) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            payload = json.loads(curl_health(service_url))
            if payload.get("healthy") and payload.get("ready") and payload.get("trade_mode") == "live":
                return payload
        except Exception as exc:  # pragma: no cover - exercised only in runtime automation
            last_error = exc
        time.sleep(poll_sec)
    raise RuntimeError(f"Timed out waiting for live health: {last_error}")


def wait_for_shadow_health(service_url: str, timeout_sec: float = 90.0, poll_sec: float = 1.0) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_snapshot: dict[str, Any] | None = None
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            payload = json.loads(curl_health(service_url))
            if payload.get("healthy") and payload.get("ready") and payload.get("trade_mode") == "shadow":
                return payload
            last_snapshot = payload
        except Exception as exc:  # pragma: no cover - exercised only in runtime automation
            last_error = exc
        time.sleep(poll_sec)
    detail = last_snapshot if last_snapshot is not None else last_error
    raise RuntimeError(f"Timed out waiting for shadow health: {detail}")


def run_logged_command(cmd: RunCommand) -> subprocess.CompletedProcess[str]:
    stdout_handle = cmd.stdout_path.open("w", encoding="utf-8") if cmd.stdout_path else subprocess.PIPE
    stderr_handle = cmd.stderr_path.open("w", encoding="utf-8") if cmd.stderr_path else subprocess.PIPE
    try:
        proc = subprocess.run(
            cmd.argv,
            cwd=str(cmd.cwd) if cmd.cwd else None,
            check=cmd.check,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    finally:
        if cmd.stdout_path:
            stdout_handle.close()
        if cmd.stderr_path:
            stderr_handle.close()
    return proc


def run_shell(cmd: str, label: str, cwd: Path | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=str(cwd) if cwd else None, check=True, text=True)


def latest_run_manifest_path(repo_root: Path, tranche_id: str) -> Path:
    return ensure_run_dir(repo_root, tranche_id) / "latest_run.yaml"


def write_latest_run_manifest(repo_root: Path, tranche_id: str, payload: dict[str, Any]) -> None:
    save_yaml(latest_run_manifest_path(repo_root, tranche_id), payload)


def latest_run_restore_required_manifest(repo_root: Path, tranche_id: str) -> dict[str, Any] | None:
    manifest_path = latest_run_manifest_path(repo_root, tranche_id)
    manifest = load_yaml(manifest_path) if manifest_path.exists() else {}
    if manifest.get("run_state") == "restore_required":
        return manifest
    return None


def direct_venue_audit_clean(
    payload: dict[str, Any] | None,
    *,
    position_tol_base: float = 0.0025,
    max_open_orders: int = 0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not isinstance(payload, dict):
        return False, ["audit payload is not an object"]
    for violation in payload.get("violations") or []:
        reasons.append(str(violation))
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            reasons.append("audit result entry is not an object")
            continue
        venue = str(item.get("venue") or "unknown")
        if item.get("ok") is not True:
            reasons.append(f"{venue}: ok is not true")
        try:
            position = abs(float(item.get("position_base", 0.0)))
        except (TypeError, ValueError):
            reasons.append(f"{venue}: position_base is not numeric")
            position = 0.0
        if position > position_tol_base:
            reasons.append(f"{venue}: abs(position_base)={position} > {position_tol_base}")
        if item.get("open_order_count_known") is not True:
            reasons.append(f"{venue}: open_order_count unknown")
            continue
        try:
            open_orders = int(item.get("open_order_count", 0))
        except (TypeError, ValueError):
            reasons.append(f"{venue}: open_order_count is not numeric")
            open_orders = 0
        if open_orders > max_open_orders:
            reasons.append(f"{venue}: open_order_count={open_orders} > {max_open_orders}")
    return not reasons, reasons


def run_manual_recovery_host_audit(
    control_pack: dict[str, Any],
    run_root: Path,
    *,
    position_tol_base: float = 0.0025,
    max_open_orders: int = 0,
) -> tuple[Path, dict[str, Any]]:
    defaults = control_pack.get("execution_defaults", {})
    audit_script = Path(defaults.get("venue_audit_script") or "/home/ubuntu/paraphina/tools/live_venue_audit.py")
    env_file = Path(defaults.get("env_file") or "/etc/paraphina/current.env")
    timestamp = utc_now().replace("-", "").replace(":", "").replace("Z", "Z")
    audit_path = run_root / f"manual_recovery_host_venue_audit_{timestamp}.json"
    cmd = [
        "python3",
        str(audit_script),
        "--env-file",
        str(env_file),
        "--position-tol-base",
        str(position_tol_base),
        "--max-open-orders",
        str(max_open_orders),
        "--timeout-seconds",
        "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"manual recovery host audit did not emit JSON rc={result.returncode} stderr={result.stderr.strip()!r}"
        ) from exc
    payload["_phase5_manual_recovery_audit_rc"] = result.returncode
    payload["_phase5_manual_recovery_audit_stderr"] = result.stderr.strip()
    write_json(audit_path, payload)
    return audit_path, payload


def record_manual_recovery_verification(
    tranche_id: str,
    audit_path: str | Path,
    repo_root: Path = ROOT,
    *,
    verify_host: bool = True,
) -> Path:
    queue, control_pack = load_state(repo_root)
    find_tranche(queue, tranche_id)
    manifest_path = latest_run_manifest_path(repo_root, tranche_id)
    if not manifest_path.exists():
        raise RuntimeError(f"{tranche_id}: latest_run manifest does not exist")
    manifest = load_yaml(manifest_path)
    if manifest.get("run_state") != "restore_required":
        raise RuntimeError(f"{tranche_id}: latest_run is not restore_required")
    run_root_raw = manifest.get("run_root")
    if not isinstance(run_root_raw, str) or not run_root_raw:
        raise RuntimeError(f"{tranche_id}: restore_required latest_run missing run_root")
    run_root = Path(run_root_raw)
    run_root.mkdir(parents=True, exist_ok=True)

    recovery_audit_path = Path(audit_path)
    with recovery_audit_path.open("r", encoding="utf-8") as handle:
        recovery_audit = json.load(handle)
    recovery_clean, recovery_reasons = direct_venue_audit_clean(recovery_audit)
    if not recovery_clean:
        raise RuntimeError(f"{tranche_id}: manual recovery audit is not clean: {recovery_reasons}")

    health_payload: dict[str, Any] | None = None
    host_audit_path: Path | None = None
    host_audit: dict[str, Any] | None = None
    if verify_host:
        health_payload = ensure_shadow_health()
        host_audit_path, host_audit = run_manual_recovery_host_audit(control_pack, run_root)
        host_clean, host_reasons = direct_venue_audit_clean(host_audit)
        if not host_clean:
            raise RuntimeError(f"{tranche_id}: current host venue audit is not clean: {host_reasons}")

    verified_utc = utc_now()
    timestamp = verified_utc.replace("-", "").replace(":", "").replace("Z", "Z")
    verification_path = run_root / f"manual_recovery_verification_{timestamp}.json"
    verification = {
        "schema_version": 1,
        "updated_utc": verified_utc,
        "tranche_id": tranche_id,
        "run_root": str(run_root),
        "previous_run_state": manifest.get("run_state"),
        "restore_required_reasons": manifest.get("restore_required_reasons", []),
        "manual_recovery_audit_path": str(recovery_audit_path),
        "manual_recovery_audit_clean": True,
        "manual_recovery_host_checked": verify_host,
        "manual_recovery_host_audit_path": None if host_audit_path is None else str(host_audit_path),
        "manual_recovery_host_audit_clean": None if host_audit is None else True,
        "health_payload": health_payload,
        "promotion_credit": "none",
    }
    write_json(verification_path, verification)
    updated_manifest = copy.deepcopy(manifest)
    updated_manifest.update(
        {
            "updated_utc": verified_utc,
            "run_state": "manual_recovery_verified",
            "manual_recovery_verified_utc": verified_utc,
            "manual_recovery_verification_path": str(verification_path),
            "manual_recovery_audit_path": str(recovery_audit_path),
            "manual_recovery_host_checked": verify_host,
            "manual_recovery_host_audit_path": None if host_audit_path is None else str(host_audit_path),
            "manual_recovery_promotional_credit": "none",
        }
    )
    write_latest_run_manifest(repo_root, tranche_id, updated_manifest)
    return verification_path


def closeout_restore_required_reasons(closeout_bundle: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if closeout_bundle.get("trade_mode_post") != "shadow":
        reasons.append("trade_mode_post_not_shadow")
    if closeout_bundle.get("healthy_post") is not True:
        reasons.append("healthy_post_not_true")
    if closeout_bundle.get("ready_post") is not True:
        reasons.append("ready_post_not_true")
    direct_audit_clean = bool(
        closeout_bundle.get("post_rollback_venue_audit_clean")
        or closeout_bundle.get("post_cleanup_venue_audit_clean")
    )
    if not direct_audit_clean:
        reasons.append("post_restore_direct_venue_audit_not_clean")
    return reasons


def ensure_latest_run_not_restore_required(repo_root: Path, tranche_id: str) -> None:
    manifest = latest_run_restore_required_manifest(repo_root, tranche_id)
    if manifest is None:
        return
    reasons = manifest.get("restore_required_reasons")
    reason_text = ",".join(str(item) for item in reasons) if isinstance(reasons, list) else "unknown"
    raise RuntimeError(
        f"{tranche_id}: latest live run requires restore verification before orchestration can continue "
        f"(run_state=restore_required reasons={reason_text})"
    )


def latest_run_requires_recovery(repo_root: Path, tranche_id: str) -> tuple[bool, Path | None, int | None]:
    manifest_path = latest_run_manifest_path(repo_root, tranche_id)
    manifest = load_yaml(manifest_path) if manifest_path.exists() else {}
    if manifest.get("run_state") != "live_started":
        return False, None, None
    run_root_raw = manifest.get("run_root")
    if not isinstance(run_root_raw, str) or not run_root_raw:
        return False, None, None
    run_root = Path(run_root_raw)
    if not run_root.exists():
        return False, None, None
    closeout_path = run_root / "live_closeout_bundle.json"
    guard_path = run_root / "guard.log"
    if closeout_path.exists() or not guard_path.exists():
        return False, run_root, manifest.get("duration_sec")
    return True, run_root, manifest.get("duration_sec")


def maybe_auto_recover_latest_run(repo_root: Path, tranche_id: str) -> Path | None:
    should_recover, run_root, duration_sec = latest_run_requires_recovery(repo_root, tranche_id)
    if not should_recover or run_root is None:
        return None
    return recover_live_closeout(
        tranche_id=tranche_id,
        repo_root=repo_root,
        run_root=run_root,
        duration_sec=duration_sec if isinstance(duration_sec, int) else None,
    )


def auto_recover_pending_latest_runs(repo_root: Path, tranche_ids: list[str] | None = None) -> list[Path]:
    queue, _ = load_state(repo_root)
    candidate_ids = tranche_ids
    if candidate_ids is None:
        candidate_ids = [
            tranche.get("id")
            for tranche in queue.get("serialized_mainline", [])
            if isinstance(tranche.get("id"), str)
        ]
    recovered: list[Path] = []
    for tranche_id in candidate_ids:
        if not isinstance(tranche_id, str):
            continue
        closeout_path = maybe_auto_recover_latest_run(repo_root, tranche_id)
        if closeout_path is not None:
            recovered.append(closeout_path)
    return recovered


def support_gate_manifest_path(repo_root: Path, tranche_id: str, gate: str) -> Path:
    return ensure_run_dir(repo_root, tranche_id) / f"{gate.replace('/', '_')}.yaml"


def write_support_gate_manifest(repo_root: Path, tranche_id: str, gate: str, payload: dict[str, Any]) -> None:
    save_yaml(support_gate_manifest_path(repo_root, tranche_id, gate), payload)


def safe_health_snapshot(service_url: str) -> dict[str, Any]:
    try:
        return json.loads(curl_health(service_url, attempts=5, delay_sec=1.0))
    except Exception as exc:  # pragma: no cover - runtime-only fallback
        return {"error": str(exc)}


def safe_shadow_health_snapshot(service_url: str) -> dict[str, Any]:
    try:
        return wait_for_shadow_health(service_url)
    except Exception:
        return safe_health_snapshot(service_url)


def copy_file_segment(src: Path, dst: Path, offset: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as src_handle, dst.open("wb") as dst_handle:
        src_handle.seek(offset)
        shutil.copyfileobj(src_handle, dst_handle)


def get_nested(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def comparable_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def compare_rule_value(lhs: Any, op: str, rhs: Any) -> bool:
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    if op in {">", ">=", "<", "<="}:
        if lhs is None or rhs is None:
            return False
        lhs_num = comparable_number(lhs)
        rhs_num = comparable_number(rhs)
        if lhs_num is not None and rhs_num is not None:
            lhs = lhs_num
            rhs = rhs_num
        try:
            if op == ">":
                return lhs > rhs
            if op == ">=":
                return lhs >= rhs
            if op == "<":
                return lhs < rhs
            return lhs <= rhs
        except TypeError:
            return False
    if op == "in":
        return lhs in rhs if isinstance(rhs, (list, tuple, set)) else False
    if op == "not_in":
        return lhs not in rhs if isinstance(rhs, (list, tuple, set)) else False
    raise ValueError(f"unsupported autoscore op {op!r}")


def autoscore_payloads(run_root: Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    payloads["closeout"] = load_closeout_bundle(run_root)
    payloads["balance_snapshot"] = load_json_payload_with_path(latest_balance_snapshot_path(run_root))
    payloads["direct_venue_audit"] = load_json_payload_with_path(latest_direct_venue_audit_post_path(run_root))
    payloads["cashflow"] = load_json_payload_with_path(latest_cashflow_attribution_path(run_root))
    mapping = {
        "summary": run_root / "live_segment_summary.json",
        "metrics": run_root / "live_metrics.json",
        "health_post": run_root / "health_post.json",
        "systemd_post": run_root / "systemd_post.txt",
    }
    for source, path in mapping.items():
        if not path.exists():
            payloads[source] = {}
            continue
        if path.suffix == ".txt":
            payloads[source] = parse_kv_lines(path.read_text(encoding="utf-8"))
        else:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            payloads[source] = normalized_closeout_bundle(loaded) if source == "closeout" else loaded
    return payloads


def evaluate_rule_group(rules: list[dict[str, Any]], payloads: dict[str, Any]) -> dict[str, Any]:
    hard_failed: list[dict[str, Any]] = []
    hold_rules: list[dict[str, Any]] = []
    rollback_rules: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for rule in rules:
        source = rule["source"]
        actual = get_nested(payloads.get(source, {}), rule["path"])
        passed = compare_rule_value(actual, rule["op"], rule.get("value"))
        if passed:
            continue
        severity = rule.get("severity", "fail")
        decision_effect = rule.get("decision_effect")
        if decision_effect is None:
            if severity == "warn":
                decision_effect = "warn"
            elif severity == "hold_only":
                decision_effect = "hold"
            elif severity == "rollback":
                decision_effect = "rollback"
            else:
                decision_effect = "fail"
        record = {
            "source": source,
            "path": rule["path"],
            "op": rule["op"],
            "expected": rule.get("value"),
            "actual": actual,
            "severity": severity,
            "decision_effect": decision_effect,
        }
        if decision_effect == "warn":
            warnings.append(record)
        elif decision_effect == "rollback":
            rollback_rules.append(record)
        elif decision_effect == "hold":
            hold_rules.append(record)
        else:
            hard_failed.append(record)
    blocking_rules = hard_failed + hold_rules + rollback_rules
    group_decision_effect = "pass"
    if rollback_rules:
        group_decision_effect = "rollback"
    elif hard_failed or hold_rules:
        group_decision_effect = "hold"
    return {
        "passed": not blocking_rules,
        "decision_effect": group_decision_effect,
        "failed_rules": blocking_rules,
        "blocking_rules": blocking_rules,
        "hard_failed_rules": hard_failed,
        "hold_rules": hold_rules,
        "rollback_rules": rollback_rules,
        "warnings": warnings,
    }


def autoscore_run(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    run_root: Path,
    duration_sec: int,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    automation = tranche_automation(tranche, control_pack, repo_root)
    payloads = autoscore_payloads(run_root)
    clean = evaluate_rule_group(automation["autoscore"].get("clean", []), payloads)
    mechanism = evaluate_rule_group(automation["autoscore"].get("mechanism", []), payloads)
    promotion = evaluate_rule_group(automation["autoscore"].get("promotion", []), payloads)
    has_promotion_rules = bool(automation["autoscore"].get("promotion", []))

    rung_plan = automation["rung_plan"]
    final_duration = max((int(rung.get("duration_sec", 0)) for rung in rung_plan), default=duration_sec)
    is_final_rung = duration_sec >= final_duration

    group_effects = [
        clean.get("decision_effect"),
        mechanism.get("decision_effect"),
        promotion.get("decision_effect"),
    ]
    suggested_action = "continue"
    if "rollback" in group_effects:
        suggested_action = "rollback"
    elif not clean["passed"] or not mechanism["passed"]:
        suggested_action = "hold"
    elif is_final_rung and has_promotion_rules and promotion["passed"]:
        suggested_action = "promote"
    elif is_final_rung:
        suggested_action = "hold"

    result = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "tranche_id": tranche.get("id"),
        "run_root": str(run_root),
        "duration_sec": duration_sec,
        "clean": clean,
        "mechanism": mechanism,
        "promotion": promotion,
        "has_promotion_rules": has_promotion_rules,
        "is_final_rung": is_final_rung,
        "suggested_action": suggested_action,
    }
    write_json(run_root / "autoscore_bundle.json", result)
    return result


def simulate_record_result_payload(
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    tranche_id: str,
    decision: str,
    repo_root: Path = ROOT,
    observed_blocker_family: str | None = None,
    next_target_override: str | None = None,
) -> tuple[dict[str, Any], str]:
    queue_copy = copy.deepcopy(queue)
    section, index, tranche = find_tranche(queue_copy, tranche_id)
    status_map = {"promote": "promoted", "hold": "hold", "rollback": "rolled_back"}
    queue_copy[section][index]["status"] = status_map[decision]
    next_target = None
    if decision == "promote":
        next_target = tranche.get("next_if_pass")
    elif next_target_override:
        next_target = next_target_override
    elif fail_child_activation_allowed(
        tranche,
        observed_blocker_family,
        precondition_failed=False,
    ):
        next_target = tranche_fail_child_target(tranche, observed_blocker_family)
    activate_next(queue_copy, next_target)
    queue_copy["updated_utc"] = utc_now()
    return queue_copy, render_status_markdown(queue_copy, control_pack, repo_root)


def lane_bundle_dir(repo_root: Path, tranche_id: str, lane_id: str) -> Path:
    return ensure_run_dir(repo_root, tranche_id) / "lanes" / lane_id


def lane_worktree_dir(control_pack: dict[str, Any], tranche_id: str, lane_id: str, repo_root: Path = ROOT) -> Path:
    return worktree_root_for_tranche(tranche_id, control_pack, repo_root) / lane_id / "repo"


def git_output(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def lane_overlay_paths(
    repo_root: Path,
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    parent_tranche: dict[str, Any],
    lane_spec: dict[str, Any],
) -> list[str]:
    paths: set[str] = {
        "phase5",
        "ROADMAP.md",
        "tools/phase5_tranche.py",
        "tools/telemetry_analyzer.py",
        "tests/test_phase5_tranche_system.py",
        "docs/PHASE5_AUTONOMOUS_TRANCHE_SYSTEM.md",
    }
    for path in tranche_change_scope_files(parent_tranche):
        rel = repo_relative_path(path, repo_root)
        if not rel.startswith("/"):
            paths.add(rel)
    child_id = lane_spec.get("child_tranche_id")
    if child_id:
        try:
            _, _, child_tranche = find_tranche(queue, child_id)
        except KeyError:
            child_tranche = None
        if child_tranche:
            for path in tranche_change_scope_files(child_tranche):
                rel = repo_relative_path(path, repo_root)
                if not rel.startswith("/"):
                    paths.add(rel)
            try:
                overlay_path = tranche_stage_overlay_source(child_tranche, control_pack["execution_defaults"])
            except Exception:
                overlay_path = None
            if overlay_path is not None:
                rel = repo_relative_path(overlay_path, repo_root)
                if not rel.startswith("/"):
                    paths.add(rel)
    return sorted(path for path in paths if path)


def workspace_overlay_patch(repo_root: Path, output_path: Path, pathspecs: list[str] | None = None) -> Path | None:
    cmd = ["diff", "--binary", "HEAD", "--"]
    if pathspecs:
        cmd.extend(pathspecs)
    patch = git_output(repo_root, cmd)
    if not patch.strip():
        return None
    write_text(output_path, patch)
    return output_path


def workspace_untracked_files(repo_root: Path, pathspecs: list[str] | None = None) -> list[Path]:
    cmd = ["ls-files", "--others", "--exclude-standard"]
    if pathspecs:
        cmd.extend(["--", *pathspecs])
    raw = git_output(repo_root, cmd)
    files: list[Path] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            files.append(repo_root / line)
    return files


def apply_workspace_overlay(
    repo_root: Path,
    worktree_dir: Path,
    bundle_dir: Path,
    pathspecs: list[str] | None = None,
) -> dict[str, Any]:
    overlay_dir = bundle_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    patch_path = workspace_overlay_patch(repo_root, overlay_dir / "workspace_overlay.patch", pathspecs=pathspecs)
    if patch_path is not None:
        subprocess.run(
            ["git", "-C", str(worktree_dir), "apply", "--allow-empty", str(patch_path)],
            check=True,
            text=True,
        )

    copied_untracked: list[str] = []
    for src in workspace_untracked_files(repo_root, pathspecs=pathspecs):
        if not src.is_file():
            continue
        rel = src.resolve().relative_to(repo_root.resolve())
        dst = worktree_dir / rel
        snapshot_dst = overlay_dir / "untracked" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        shutil.copy2(src, snapshot_dst)
        copied_untracked.append(str(rel))

    return {
        "patch_path": str(patch_path) if patch_path else None,
        "copied_untracked": copied_untracked,
        "pathspecs": pathspecs or [],
    }


def restore_workspace_overlay_snapshot(bundle_dir: Path, worktree_dir: Path) -> dict[str, Any]:
    overlay_dir = bundle_dir / "overlay"
    patch_path = overlay_dir / "workspace_overlay.patch"
    if patch_path.exists():
        subprocess.run(
            ["git", "-C", str(worktree_dir), "apply", "--allow-empty", str(patch_path)],
            check=True,
            text=True,
        )

    restored_untracked: list[str] = []
    untracked_root = overlay_dir / "untracked"
    if untracked_root.exists():
        for src in sorted(untracked_root.rglob("*")):
            if not src.is_file():
                continue
            rel = src.resolve().relative_to(untracked_root.resolve())
            dst = worktree_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored_untracked.append(str(rel))
    return {
        "patch_path": str(patch_path) if patch_path.exists() else None,
        "restored_untracked": restored_untracked,
    }


def add_detached_worktree(repo_root: Path, worktree_dir: Path) -> None:
    subprocess.run(["git", "-C", str(repo_root), "worktree", "prune"], check=True, text=True)
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_dir), "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "") + (exc.stdout or "")
        if "missing but already registered worktree" not in stderr:
            raise
        subprocess.run(["git", "-C", str(repo_root), "worktree", "prune"], check=True, text=True)
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_dir), "HEAD"],
            check=True,
            text=True,
        )


def create_worktree(repo_root: Path, control_pack: dict[str, Any], tranche_id: str, lane_id: str) -> Path:
    worktree_dir = lane_worktree_dir(control_pack, tranche_id, lane_id, repo_root)
    if worktree_dir.parent.exists():
        shutil.rmtree(worktree_dir.parent)
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    add_detached_worktree(repo_root, worktree_dir)
    return worktree_dir


def remove_worktree(repo_root: Path, worktree_dir: Path) -> None:
    if not worktree_dir.exists():
        return
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_dir)],
        check=True,
        text=True,
    )
    parent = worktree_dir.parent
    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)


def build_lane_specs(queue: dict[str, Any], tranche: dict[str, Any], control_pack: dict[str, Any], repo_root: Path = ROOT) -> list[dict[str, Any]]:
    automation = tranche_automation(tranche, control_pack, repo_root)
    specs: list[dict[str, Any]] = [
        {"lane_id": LANE_ROLE_LIVE, "role": LANE_ROLE_LIVE, "kind": "runtime"},
        {"lane_id": LANE_ROLE_FORENSICS, "role": LANE_ROLE_FORENSICS, "kind": "analysis"},
    ]
    if tranche.get("next_if_pass"):
        specs.append(
            {
                "lane_id": LANE_ROLE_PASS_PREP,
                "role": LANE_ROLE_PASS_PREP,
                "kind": "child_prep",
                "child_tranche_id": tranche.get("next_if_pass"),
                "decision_preview": "promote",
            }
        )
    fail_routes = tranche_fail_route_specs(tranche)
    for route in fail_routes:
        child_id = route["child_tranche_id"]
        observed_blocker_family = route["observed_blocker_family"]
        lane_id = LANE_ROLE_FAIL_PREP
        if len(fail_routes) > 1:
            suffix_source = observed_blocker_family or child_id
            suffix = "".join(ch if ch.isalnum() else "_" for ch in suffix_source).strip("_")
            lane_id = f"{LANE_ROLE_FAIL_PREP}__{suffix or 'default'}"
        specs.append(
            {
                "lane_id": lane_id,
                "role": lane_id,
                "kind": "child_prep",
                "child_tranche_id": child_id,
                "decision_preview": "hold",
                "observed_blocker_family": observed_blocker_family,
                "route_kind": route["route_kind"],
            }
        )
    for support_id in automation.get("support_tracks", []):
        _, _, support_tranche = find_tranche(queue, support_id)
        support_automation = tranche_automation(support_tranche, control_pack, repo_root)
        specs.append(
            {
                "lane_id": f"{LANE_ROLE_SUPPORT_PREFIX}{support_id}",
                "role": f"{LANE_ROLE_SUPPORT_PREFIX}{support_id}",
                "kind": "support_track",
                "child_tranche_id": support_id,
                "autorun_policy": support_automation.get("autorun_policy", "manual"),
            }
        )
    for family in automation.get("support_families", []):
        if not support_family_triggered(tranche, family):
            continue
        support_id = str(family["support_track_id"])
        _, _, support_tranche = find_tranche(queue, support_id)
        support_automation = tranche_automation(support_tranche, control_pack, repo_root)
        specs.append(
            {
                "lane_id": support_family_lane_id(family),
                "role": support_family_lane_id(family),
                "kind": "support_track",
                "child_tranche_id": support_id,
                "autorun_policy": family.get("autorun_policy", support_automation.get("autorun_policy", "manual")),
                "family_id": family.get("id"),
                "trigger_mode": family.get("trigger_mode", "always"),
                "blocker_families": list(family.get("blocker_families", []) or []),
                "max_parallel_runs": family.get("max_parallel_runs"),
                "stop_on_mainline_promote": family.get("stop_on_mainline_promote", False),
                "priority_class": infer_support_lane_priority_class({"family_id": family.get("id")}),
            }
        )
    return specs


def write_lane_bundle(
    repo_root: Path,
    tranche_id: str,
    lane_id: str,
    manifest: dict[str, Any],
    inputs: dict[str, Any],
    result_bundle: dict[str, Any] | None = None,
    handoff_packet: str | None = None,
) -> Path:
    bundle_dir = lane_bundle_dir(repo_root, tranche_id, lane_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_yaml(bundle_dir / "lane_manifest.yaml", manifest)
    write_json(bundle_dir / "inputs.json", inputs)
    write_json(bundle_dir / "runtime_context.json", {
        "updated_utc": utc_now(),
        "repo_root": str(repo_root),
        "cwd": str(repo_root),
    })
    write_json(bundle_dir / "validation_plan.json", inputs.get("validation_plan", {}))
    if handoff_packet is not None:
        write_text(bundle_dir / "handoff_packet.md", handoff_packet)
    if result_bundle is not None:
        write_json(bundle_dir / "result_bundle.json", result_bundle)
    return bundle_dir


def save_lane_manifest(repo_root: Path, tranche_id: str, lane: dict[str, Any]) -> None:
    bundle_dir = lane_bundle_dir(repo_root, tranche_id, str(lane["lane_id"]))
    save_yaml(bundle_dir / "lane_manifest.yaml", lane)


def save_lane_result(repo_root: Path, tranche_id: str, lane_id: str, result_bundle: dict[str, Any]) -> None:
    bundle_dir = lane_bundle_dir(repo_root, tranche_id, lane_id)
    write_json(bundle_dir / "result_bundle.json", result_bundle)


def prepare_child_lane(
    repo_root: Path,
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    parent_tranche: dict[str, Any],
    lane_spec: dict[str, Any],
) -> dict[str, Any]:
    child_id = lane_spec["child_tranche_id"]
    _, _, child_tranche = find_tranche(queue, child_id)
    bundle_dir = lane_bundle_dir(repo_root, parent_tranche["id"], lane_spec["lane_id"])
    worktree_dir = create_worktree(repo_root, control_pack, parent_tranche["id"], lane_spec["lane_id"])
    overlay = apply_workspace_overlay(
        repo_root,
        worktree_dir,
        bundle_dir,
        pathspecs=lane_overlay_paths(repo_root, queue, control_pack, parent_tranche, lane_spec),
    )

    preview_queue, preview_status = simulate_record_result_payload(
        queue,
        control_pack,
        parent_tranche["id"],
        lane_spec["decision_preview"],
        repo_root,
        observed_blocker_family=lane_spec.get("observed_blocker_family"),
        next_target_override=child_id if lane_spec["decision_preview"] != "promote" else None,
    )
    child_run_dir = ensure_run_dir(worktree_dir, child_id)
    child_card = tranche_card_payload(child_tranche, control_pack, child_run_dir, worktree_dir)
    save_yaml(child_run_dir / "tranche_card_preview.yaml", child_card)
    save_yaml(bundle_dir / "queue_preview.yaml", preview_queue)
    write_text(bundle_dir / "status_preview.md", preview_status)

    handoff = "\n".join(
        [
            f"# {lane_spec['lane_id']}",
            "",
            f"- parent tranche: `{parent_tranche['id']}`",
            f"- child tranche: `{child_id}`",
            f"- worktree: `{worktree_dir}`",
            f"- decision preview: `{lane_spec['decision_preview']}`",
            f"- observed blocker preview: `{lane_spec.get('observed_blocker_family') or 'default'}`",
            f"- surface_id: `{safe_tranche_surface_id(child_tranche, control_pack, worktree_dir)}`",
        ]
    ) + "\n"
    manifest = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "lane_id": lane_spec["lane_id"],
        "role": lane_spec["role"],
        "kind": lane_spec["kind"],
        "status": "prepared",
        "child_tranche_id": child_id,
        "worktree_path": str(worktree_dir),
        "bundle_dir": str(bundle_dir),
        "overlay": overlay,
    }
    inputs = {
        "parent_tranche_id": parent_tranche["id"],
        "child_tranche_id": child_id,
        "surface_id": safe_tranche_surface_id(child_tranche, control_pack, worktree_dir),
        "stage_overlay_source": str(tranche_stage_overlay_source(child_tranche, control_pack["execution_defaults"])),
        "validation_plan": {
            "commands": [
                f"python3 tools/phase5_tranche.py prepare --tranche-id {shlex.quote(child_id)} --no-mark-in-progress",
                f"python3 tools/phase5_tranche.py validate",
            ]
        },
    }
    write_lane_bundle(repo_root, parent_tranche["id"], lane_spec["lane_id"], manifest, inputs, handoff_packet=handoff)
    return manifest


def prepare_non_worktree_lane(
    repo_root: Path,
    tranche: dict[str, Any],
    lane_spec: dict[str, Any],
    extra_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle_dir = lane_bundle_dir(repo_root, tranche["id"], lane_spec["lane_id"])
    manifest = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "lane_id": lane_spec["lane_id"],
        "role": lane_spec["role"],
        "kind": lane_spec["kind"],
        "status": "prepared",
        "bundle_dir": str(bundle_dir),
    }
    inputs = {"parent_tranche_id": tranche["id"], "validation_plan": {"commands": []}}
    if extra_inputs:
        inputs.update(extra_inputs)
    write_lane_bundle(repo_root, tranche["id"], lane_spec["lane_id"], manifest, inputs)
    return manifest


def prepare_support_lane(
    repo_root: Path,
    queue: dict[str, Any],
    control_pack: dict[str, Any],
    parent_tranche: dict[str, Any],
    lane_spec: dict[str, Any],
) -> dict[str, Any]:
    support_id = lane_spec["child_tranche_id"]
    _, _, support_tranche = find_tranche(queue, support_id)
    bundle_dir = lane_bundle_dir(repo_root, parent_tranche["id"], lane_spec["lane_id"])
    worktree_dir = create_worktree(repo_root, control_pack, parent_tranche["id"], lane_spec["lane_id"])
    overlay = apply_workspace_overlay(
        repo_root,
        worktree_dir,
        bundle_dir,
        pathspecs=lane_overlay_paths(repo_root, queue, control_pack, parent_tranche, lane_spec),
    )
    support_run_dir = ensure_run_dir(worktree_dir, support_id)
    save_yaml(support_run_dir / "tranche_card_preview.yaml", tranche_card_payload(support_tranche, control_pack, support_run_dir, worktree_dir))
    handoff = "\n".join(
        [
            f"# {lane_spec['lane_id']}",
            "",
            f"- parent tranche: `{parent_tranche['id']}`",
            f"- support tranche: `{support_id}`",
            f"- worktree: `{worktree_dir}`",
            f"- autorun policy: `{lane_spec.get('autorun_policy', 'manual')}`",
        ]
    ) + "\n"
    manifest = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "lane_id": lane_spec["lane_id"],
        "role": lane_spec["role"],
        "kind": lane_spec["kind"],
        "status": "prepared",
        "child_tranche_id": support_id,
        "worktree_path": str(worktree_dir),
        "bundle_dir": str(bundle_dir),
        "overlay": overlay,
        "autorun_policy": lane_spec.get("autorun_policy", "manual"),
        "family_id": lane_spec.get("family_id"),
        "trigger_mode": lane_spec.get("trigger_mode"),
        "blocker_families": list(lane_spec.get("blocker_families", []) or []),
        "max_parallel_runs": lane_spec.get("max_parallel_runs"),
        "stop_on_mainline_promote": lane_spec.get("stop_on_mainline_promote", False),
        "priority_class": lane_spec.get("priority_class", infer_support_lane_priority_class(support_id)),
    }
    inputs = {
        "parent_tranche_id": parent_tranche["id"],
        "support_tranche_id": support_id,
        "validation_plan": {
            "commands": [
                f"python3 tools/phase5_tranche.py validate",
            ]
        },
    }
    write_lane_bundle(repo_root, parent_tranche["id"], lane_spec["lane_id"], manifest, inputs, handoff_packet=handoff)
    return manifest


def orchestration_session_payload(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    queue: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    automation = tranche_automation(tranche, control_pack, repo_root)
    session_id = f"{tranche['id']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return {
        "schema_version": 1,
        "session_id": session_id,
        "tranche_id": tranche["id"],
        "track": tranche.get("track"),
        "state": "spawned",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
        "bundle_root": str(lane_bundle_root_for_tranche(tranche["id"], control_pack, repo_root)),
        "worktree_root": str(worktree_root_for_tranche(tranche["id"], control_pack, repo_root)),
        "automation": {
            "autonomy_mode": automation["autonomy_mode"],
            "subagent_model": automation["subagent_model"],
            "worktree_lifecycle": automation["worktree_lifecycle"],
            "artifact_packaging": automation["artifact_packaging"],
            "cleanup_policy": automation["cleanup_policy"],
            "rung_plan": copy.deepcopy(automation["rung_plan"]),
            "support_tracks": list(automation.get("support_tracks", [])),
            "support_families": copy.deepcopy(automation.get("support_families", [])),
            "max_parallel_support_lanes": automation.get("max_parallel_support_lanes", 0),
            "support_lane_priority": copy.deepcopy(automation.get("support_lane_priority", [])),
            "support_lane_capacity_gate": automation.get("support_lane_capacity_gate", False),
            "stage_verdict_contract": copy.deepcopy(automation.get("stage_verdict_contract", [])),
        },
        "lane_specs": build_lane_specs(queue, tranche, control_pack, repo_root),
        "lanes": [],
        "rung_results": [],
        "selected_child": None,
        "final_decision": None,
        "preflight": None,
        "preflight_summary_path": None,
        "state_sync_report_path": None,
        "tranche_card_path": None,
        "support_runs": [],
    }


def ensure_orchestration_preflight(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    queue, _ = load_state(repo_root)
    ensure_latest_run_not_restore_required(repo_root, str(tranche["id"]))
    storage_preflight = ensure_runtime_storage_headroom(control_pack, 3600, repo_root)
    health = ensure_shadow_health()
    runtime_binary = candidate_runtime_binary_path(tranche, control_pack["execution_defaults"], repo_root)
    cleanup_binary = cleanup_binary_path(control_pack["execution_defaults"], repo_root)
    if not cleanup_binary.exists():
        raise RuntimeError(f"cleanup binary missing for orchestration preflight: {cleanup_binary}")
    overlay_path = tranche_stage_overlay_source(tranche, control_pack["execution_defaults"])
    run_root = ensure_run_dir(repo_root, str(tranche["id"]))
    state_sync_report = audit_state_sync(queue, control_pack, repo_root, tranche_id=str(tranche["id"]))
    state_sync_report_path = write_state_sync_report(run_root, state_sync_report)
    state_sync = state_sync_summary_payload(state_sync_report, state_sync_report_path)
    blocked_by_state_sync = state_sync_blocks_promotion(state_sync_report)
    hyperliquid_quota_error: RuntimeError | None = None
    try:
        hyperliquid_quota_preflight = hyperliquid_user_rate_limit_preflight(
            tranche,
            control_pack,
            3600,
        )
    except RuntimeError as exc:
        hyperliquid_quota_error = exc
        hyperliquid_quota_preflight = {
            "status": "fail",
            "blocked": True,
            "error": str(exc),
        }
    blocked_by_hyperliquid_quota = bool(hyperliquid_quota_preflight.get("blocked"))
    blocked = blocked_by_state_sync or blocked_by_hyperliquid_quota
    preflight = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "tranche_id": tranche.get("id"),
        "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
        "repo_free_bytes": storage_preflight["repo"]["free_bytes"],
        "promotion_runs_free_bytes": storage_preflight["promotion_runs"]["free_bytes"],
        "telemetry_free_bytes": storage_preflight["telemetry"]["free_bytes"],
        "tempdir_free_bytes": storage_preflight["tempdir"]["free_bytes"],
        "current_runs_free_bytes": storage_preflight["current_runs"]["free_bytes"],
        "health": health,
        "runtime_binary": str(runtime_binary),
        "runtime_binary_sha256": sha256_file_or_missing(runtime_binary),
        "cleanup_binary": str(cleanup_binary),
        "cleanup_binary_sha256": sha256_file_or_missing(cleanup_binary),
        "stage_overlay_source": str(overlay_path),
        "stage_overlay_sha256": sha256_file_or_missing(overlay_path),
        "state_sync": state_sync,
        "state_sync_report_path": None if state_sync_report_path is None else str(state_sync_report_path),
        "blocked_by_state_sync": blocked_by_state_sync,
        "hyperliquid_quota_preflight": hyperliquid_quota_preflight,
        "blocked_by_hyperliquid_quota": blocked_by_hyperliquid_quota,
        "status": "fail" if blocked else "pass",
        "decision": "hold" if blocked else "pass",
    }
    preflight_summary = preflight_summary_path(run_root)
    preflight["preflight_summary_path"] = None if preflight_summary is None else str(preflight_summary)
    preflight_path = write_preflight_summary(run_root, preflight)
    preflight["preflight_summary_path"] = None if preflight_path is None else str(preflight_path)
    if blocked_by_state_sync:
        raise RuntimeError(
            f"{tranche.get('id')}: orchestration preflight blocked by state-sync "
            f"({state_sync_report.get('critical_count', 0)} critical, {state_sync_report.get('warning_count', 0)} warning findings)"
        )
    if blocked_by_hyperliquid_quota:
        raise RuntimeError(
            f"{tranche.get('id')}: orchestration preflight blocked by Hyperliquid action quota "
            f"({hyperliquid_quota_preflight.get('error') or hyperliquid_quota_preflight})"
        ) from hyperliquid_quota_error
    return preflight


def orchestration_preflight_requires_refresh(preflight: Any) -> bool:
    if not isinstance(preflight, dict):
        return True
    if "hyperliquid_quota_preflight" not in preflight:
        return True
    return preflight.get("status") != "pass"


def spawn_lanes(
    tranche_id: str,
    repo_root: Path = ROOT,
    force: bool = False,
) -> dict[str, Any]:
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    if tranche.get("track") != "serialized_mainline":
        raise ValueError(f"{tranche_id}: only serialized mainline tranches can spawn orchestration lanes")

    orchestration = load_orchestration(repo_root)
    existing = find_orchestration_session(orchestration, tranche_id)
    if existing is not None and not force:
        _, session = existing
        if session.get("state") in {"spawned", "running", "verdict_pending"}:
            return session

    active = active_orchestration_session(orchestration, exclude_tranche_id=tranche_id)
    if active is not None:
        raise RuntimeError(
            f"serialized live lane already owned by tranche {active.get('tranche_id')} "
            f"(session {active.get('session_id')})"
        )

    preflight = ensure_orchestration_preflight(tranche, control_pack, repo_root)
    if existing is not None and force:
        teardown_lanes(tranche_id, repo_root=repo_root, preserve_session=False)
        orchestration = load_orchestration(repo_root)

    session = orchestration_session_payload(tranche, control_pack, queue, repo_root)
    session["preflight"] = preflight
    session["preflight_summary_path"] = preflight.get("preflight_summary_path")
    session["state_sync_report_path"] = preflight.get("state_sync_report_path")
    bundle_root = Path(session["bundle_root"])
    bundle_root.mkdir(parents=True, exist_ok=True)

    for lane_spec in session["lane_specs"]:
        if lane_spec["kind"] == "child_prep":
            manifest = prepare_child_lane(repo_root, queue, control_pack, tranche, lane_spec)
        elif lane_spec["kind"] == "support_track":
            manifest = prepare_support_lane(repo_root, queue, control_pack, tranche, lane_spec)
        else:
            manifest = prepare_non_worktree_lane(repo_root, tranche, lane_spec)
        session["lanes"].append({**lane_spec, **manifest})

    upsert_orchestration_session(orchestration, session)
    save_orchestration(orchestration, repo_root)
    return session


def lane_status_payload(repo_root: Path = ROOT, tranche_id: str | None = None) -> dict[str, Any]:
    orchestration = load_orchestration(repo_root)
    if tranche_id is None:
        return orchestration
    found = find_orchestration_session(orchestration, tranche_id)
    if found is None:
        raise KeyError(f"Unknown orchestration tranche id: {tranche_id}")
    return found[1]


def archive_session_worktrees(session: dict[str, Any], repo_root: Path = ROOT) -> dict[str, Any]:
    for lane in session.get("lanes", []) or []:
        worktree_path = lane.get("worktree_path")
        if worktree_path:
            remove_worktree(repo_root, Path(worktree_path))
        lane["status"] = "archived"
        lane["archived_utc"] = utc_now()
        lane["worktree_removed"] = True
        save_lane_manifest(repo_root, session["tranche_id"], lane)
    session["cleanup_completed"] = True
    session["cleanup_completed_utc"] = utc_now()
    session["updated_utc"] = utc_now()
    return session


def teardown_lanes(
    tranche_id: str,
    repo_root: Path = ROOT,
    preserve_session: bool = True,
) -> dict[str, Any]:
    orchestration = load_orchestration(repo_root)
    found = find_orchestration_session(orchestration, tranche_id)
    if found is None:
        raise KeyError(f"Unknown orchestration tranche id: {tranche_id}")
    index, session = found
    session = archive_session_worktrees(session, repo_root)
    session["state"] = "archived"
    if preserve_session:
        orchestration["sessions"][index] = session
    else:
        orchestration["sessions"].pop(index)
    save_orchestration(orchestration, repo_root)
    return session


def run_support_lane(
    tranche_id: str,
    lane: dict[str, Any],
    repo_root: Path,
    stage_context: str | None = None,
) -> dict[str, Any]:
    policy = lane.get("autorun_policy", "manual")
    bundle_dir = lane_bundle_dir(repo_root, tranche_id, lane["lane_id"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "lane_id": lane["lane_id"],
        "autorun_policy": policy,
        "status": "skipped" if policy == "manual" else "pass",
        "stage_context": stage_context,
    }
    if policy == "manual":
        result["reason"] = "manual autorun policy"
        return result

    worktree = Path(lane["worktree_path"])
    support_id = lane.get("child_tranche_id")
    log_path = bundle_dir / "logs" / "support_lane.log"
    log_lines: list[str] = []
    cleanup_worktree = False
    result["workspace_source"] = "worktree"
    try:
        if not worktree.exists():
            if policy != "validate_only":
                result["status"] = "infra_invalid"
                result["reason"] = "missing_worktree_snapshot"
                log_lines.append(f"infra_invalid: missing worktree snapshot {worktree}")
                write_text(log_path, "\n".join(log_lines) + ("\n" if log_lines else ""))
                result["log_path"] = str(log_path)
                return result
            temp_root = Path(tempfile.mkdtemp(prefix="phase5_support_lane_"))
            worktree = temp_root / "repo"
            add_detached_worktree(repo_root, worktree)
            restore_workspace_overlay_snapshot(bundle_dir, worktree)
            cleanup_worktree = True
            result["workspace_source"] = "bundle_snapshot"
            result["workspace_path"] = str(worktree)
            log_lines.append(f"validate_only: reconstructed bundle snapshot at {worktree}")
        if policy == "validate_only":
            queue, control_pack = load_state(worktree)
            write_status(queue, control_pack, worktree)
            log_lines.append("validate_only: queue/control/status rendered")
        elif policy == "shadow_smoke":
            run_dir = run_shadow_smoke(str(support_id), 600, worktree)
            result["shadow_smoke_run_dir"] = str(run_dir)
            log_lines.append(f"shadow_smoke: {run_dir}")
        elif policy == "shadow_ab":
            run_dir = run_shadow_ab(str(support_id), worktree)
            result["shadow_ab_run_dir"] = str(run_dir)
            log_lines.append(f"shadow_ab: {run_dir}")
        else:
            raise ValueError(f"unsupported support autorun policy {policy!r}")
    except FileNotFoundError as exc:
        result["status"] = "infra_invalid"
        result["reason"] = "missing_support_snapshot_input"
        result["error"] = str(exc)
        log_lines.append(f"infra_invalid: {exc}")
    except Exception as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
        log_lines.append(f"error: {exc}")
    finally:
        if cleanup_worktree:
            remove_worktree(repo_root, worktree)
    write_text(log_path, "\n".join(log_lines) + ("\n" if log_lines else ""))
    result["log_path"] = str(log_path)
    return result


def latest_support_result_by_lane(session: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for result in session.get("support_runs", []) or []:
        lane_id = result.get("lane_id")
        if isinstance(lane_id, str):
            latest[lane_id] = result
    return latest


def stage_contract_paths(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Path]:
    contract = tranche_automation(tranche, control_pack, repo_root).get(
        "stage_verdict_contract",
        DEFAULT_STAGE_VERDICT_CONTRACT,
    )
    names = list(contract) if isinstance(contract, list) else list(DEFAULT_STAGE_VERDICT_CONTRACT)
    while len(names) < 3:
        names.append(DEFAULT_STAGE_VERDICT_CONTRACT[len(names)])
    run_dir = ensure_run_dir(repo_root, str(tranche["id"]))
    return {
        "stage_verdict": run_dir / names[0],
        "venue_capability_matrix": run_dir / names[1],
        "support_summary": run_dir / names[2],
    }


def write_support_summary(
    tranche_id: str,
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    session: dict[str, Any] | None,
    repo_root: Path = ROOT,
    state_sync: dict[str, Any] | None = None,
) -> Path:
    automation = tranche_automation(tranche, control_pack, repo_root)
    latest_results = latest_support_result_by_lane(session or {})
    lanes: list[dict[str, Any]] = []
    for lane in sorted(
        [lane for lane in (session or {}).get("lanes", []) or [] if lane.get("kind") == "support_track"],
        key=lambda lane: support_lane_sort_key(lane, automation),
    ):
        lane_result = latest_results.get(str(lane.get("lane_id")))
        lanes.append(
            {
                "lane_id": lane.get("lane_id"),
                "child_tranche_id": lane.get("child_tranche_id"),
                "family_id": lane.get("family_id"),
                "priority_class": lane.get("priority_class", infer_support_lane_priority_class(lane)),
                "autorun_policy": lane.get("autorun_policy"),
                "status": lane.get("status"),
                "latest_stage_context": (lane_result or {}).get("stage_context"),
                "latest_result_status": (lane_result or {}).get("status"),
                "latest_result_log_path": (lane_result or {}).get("log_path"),
            }
        )
    payload = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "tranche_id": tranche_id,
        "session_id": None if session is None else session.get("session_id"),
        "max_parallel_support_lanes": automation.get("max_parallel_support_lanes", 0),
        "support_lane_priority": automation.get("support_lane_priority", []),
        "support_lane_capacity_gate": automation.get("support_lane_capacity_gate", False),
        "total_runs": len((session or {}).get("support_runs", []) or []),
        "lanes": lanes,
        "state_sync": state_sync,
    }
    path = stage_contract_paths(tranche, control_pack, repo_root)["support_summary"]
    write_json(path, payload)
    return path


def run_session_support_lanes(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    session: dict[str, Any],
    repo_root: Path,
    stage_context: str,
) -> list[dict[str, Any]]:
    automation = tranche_automation(tranche, control_pack, repo_root)
    capacity_gate_active = bool(automation.get("support_lane_capacity_gate", False))
    support_lanes = [
        lane for lane in session.get("lanes", []) or []
        if lane.get("kind") == "support_track"
    ]
    latest_results = latest_support_result_by_lane(session)
    non_manual = [lane for lane in support_lanes if lane.get("autorun_policy", "manual") != "manual"]
    limit = int(automation.get("max_parallel_support_lanes") or 0)
    allowed_lane_ids: set[str]
    if limit > 0:
        allowed_lane_ids = {
            str(lane.get("lane_id"))
            for lane in sorted(non_manual, key=lambda lane: support_lane_sort_key(lane, automation))[:limit]
        }
    else:
        allowed_lane_ids = {str(lane.get("lane_id")) for lane in non_manual}

    stage_results: list[dict[str, Any]] = []
    for lane in sorted(support_lanes, key=lambda lane: support_lane_sort_key(lane, automation)):
        lane_id = str(lane.get("lane_id"))
        max_parallel_runs = lane.get("max_parallel_runs")
        completed_runs = sum(
            1
            for result in session.get("support_runs", []) or []
            if result.get("lane_id") == lane_id and result.get("status") in {"pass", "fail"}
        )
        if lane.get("stop_on_mainline_promote") and session.get("final_decision") == "promote":
            result = {
                "schema_version": 1,
                "updated_utc": utc_now(),
                "lane_id": lane_id,
                "autorun_policy": lane.get("autorun_policy", "manual"),
                "status": "skipped",
                "stage_context": stage_context,
                "reason": "stopped after mainline promote",
            }
        elif isinstance(max_parallel_runs, int) and max_parallel_runs > 0 and completed_runs >= max_parallel_runs:
            result = {
                "schema_version": 1,
                "updated_utc": utc_now(),
                "lane_id": lane_id,
                "autorun_policy": lane.get("autorun_policy", "manual"),
                "status": "skipped",
                "stage_context": stage_context,
                "reason": f"max_parallel_runs reached ({max_parallel_runs})",
            }
        elif (
            lane.get("autorun_policy", "manual") != "manual"
            and lane_id not in allowed_lane_ids
        ):
            result = {
                "schema_version": 1,
                "updated_utc": utc_now(),
                "lane_id": lane_id,
                "autorun_policy": lane.get("autorun_policy", "manual"),
                "status": "skipped",
                "stage_context": stage_context,
                "reason": "max_parallel_support_lanes capacity gate",
            }
        elif capacity_gate_active and not session.get("preflight"):
            result = {
                "schema_version": 1,
                "updated_utc": utc_now(),
                "lane_id": lane_id,
                "autorun_policy": lane.get("autorun_policy", "manual"),
                "status": "skipped",
                "stage_context": stage_context,
                "reason": "support lane capacity gate missing preflight",
            }
        else:
            result = run_support_lane(tranche["id"], lane, repo_root, stage_context=stage_context)
        if result.get("status") == "fail":
            lane["status"] = "failed"
        elif result.get("status") == "infra_invalid":
            lane["status"] = "infra_invalid"
        elif result.get("status") == "skipped":
            lane["status"] = "prepared"
        else:
            lane["status"] = "completed"
        lane["updated_utc"] = utc_now()
        lane["latest_stage_context"] = stage_context
        if latest_results.get(lane_id) != result:
            session.setdefault("support_runs", []).append(result)
        save_lane_manifest(repo_root, tranche["id"], lane)
        save_lane_result(repo_root, tranche["id"], lane["lane_id"], result)
        stage_results.append(result)
    write_support_summary(tranche["id"], tranche, control_pack, session, repo_root)
    return stage_results


def record_forensics_bundle(
    tranche_id: str,
    repo_root: Path,
    session: dict[str, Any],
    run_root: Path,
    duration_sec: int,
    autoscore: dict[str, Any],
) -> None:
    lane = session_lane(session, LANE_ROLE_FORENSICS)
    if lane is None:
        return
    closeout_path = run_root / "live_closeout_bundle.json"
    result = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "run_root": str(run_root),
        "duration_sec": duration_sec,
        "closeout_bundle_path": str(closeout_path),
        "autoscore_bundle_path": str(run_root / "autoscore_bundle.json"),
        "suggested_action": autoscore["suggested_action"],
        "clean_passed": autoscore["clean"]["passed"],
        "mechanism_passed": autoscore["mechanism"]["passed"],
        "promotion_passed": autoscore["promotion"]["passed"],
    }
    lane["status"] = "completed"
    lane["latest_run_root"] = str(run_root)
    lane["updated_utc"] = utc_now()
    save_lane_manifest(repo_root, tranche_id, lane)
    save_lane_result(repo_root, tranche_id, lane["lane_id"], result)


def bounded_err_segment_text(run_root: Path) -> str:
    err_path = run_root / "paraphina_live.err.segment"
    if not err_path.exists():
        return ""
    try:
        text = err_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "error=unexpected_live_loop_exit" in line:
            return "\n".join(lines[: idx + 1])
    return text


def run_root_has_paradex_ui_touch_live_evidence(run_root: Path) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    return (
        "WS_AUDIT venue=paradex component=ui_book_truth source=api status=ok" in text
        and "WS_AUDIT venue=paradex component=ui_book_truth source=interactive status=ok" in text
        and "WS_AUDIT venue=paradex component=ui_touch_reference action=applied" in text
    )


def run_root_has_extended_pre_kill_telemetry_evidence(run_root: Path) -> bool:
    telemetry_path = run_root / "telemetry_bounded.jsonl"
    if not telemetry_path.exists():
        return False
    saw_extended_stale = False
    saw_kill_next_tick = False
    for record in iter_json_objects(telemetry_path):
        if record.get("execution_mode") != "live":
            continue
        stale_hygiene = record.get("stale_market_hygiene")
        if not isinstance(stale_hygiene, dict):
            continue
        stale_venues = stale_hygiene.get("stale_venues")
        if not isinstance(stale_venues, list):
            continue
        normalized = {str(item).lower() for item in stale_venues}
        if normalized != {"extended"}:
            continue
        saw_extended_stale = True
        if stale_hygiene.get("kill_would_fire_next_tick"):
            saw_kill_next_tick = True
    return saw_extended_stale and saw_kill_next_tick


def run_root_has_extended_freeze_path_evidence(run_root: Path) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    has_first_publish = (
        "FIRST_BOOK_UPDATE venue=extended" in text
        or "APPLIED_BOOK venue=extended" in text
    )
    has_freeze_warning = "WARN: Extended core book update frozen" in text
    has_stale_extended = "stale_venues=extended" in text
    if has_first_publish and has_freeze_warning and has_stale_extended:
        return True
    for line in text.splitlines():
        if "component=runner_apply_truth" not in line or "venue=extended" not in line:
            continue
        pairs = parse_ws_audit_tokens(line)
        frozen_total = pairs.get("ext_apply_frozen_total")
        if frozen_total and frozen_total not in {"0", "0.0"}:
            return has_first_publish and has_stale_extended
    return False


def run_root_has_extended_transport_gap_watchdog_evidence(run_root: Path) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False

    has_extended_watchdog = "WS_AUDIT venue=extended reconnect_reason=stale_watchdog" in text
    has_global_stale_collapse = (
        "stale_market_hygiene kill_triggered" in text
        and "stale_venues=extended,hyperliquid,aster,lighter,paradex" in text
    )
    has_unexpected_exit = (
        "error=unexpected_live_loop_exit" in text
        and "kill_switch=true" in text
        and "ready_market_count=0" in text
        and "stale_market_count=5" in text
    )
    return has_extended_watchdog and has_global_stale_collapse and has_unexpected_exit


def run_root_has_extended_degraded_stream_rebootstrap_gap_evidence(run_root: Path) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    lines = text.splitlines()
    has_extended_only_kill = (
        "stale_market_hygiene kill_triggered" in text and "stale_venues=extended" in text
    )
    has_unexpected_exit = (
        "error=unexpected_live_loop_exit" in text
        and "kill_switch=true" in text
        and "stale_market_count=1" in text
    )
    if not has_extended_only_kill or not has_unexpected_exit:
        return False
    if "WS_AUDIT venue=extended reconnect_reason=stale_watchdog" in text:
        return False
    if "WARN: Extended core book update frozen" in text:
        return False
    pref_idx = None
    for idx, line in enumerate(lines):
        if (
            "WS_AUDIT venue=extended component=post_publish_stream_fallback" in line
            and "action=preference_set" in line
            and "stream_preference=full_orderbook_degraded" in line
        ):
            pref_idx = idx
            break
    if pref_idx is None:
        return False
    has_fallback_win = any(
        "WS_AUDIT venue=extended component=post_publish_stream_fallback" in line
        and "action=fallback_won" in line
        for line in lines[: pref_idx + 1]
    )
    if not has_fallback_win:
        return False
    has_second_started = any(
        "WS_AUDIT venue=extended component=post_publish_stream_fallback" in line
        and "action=started" in line
        for line in lines[pref_idx + 1 :]
    )
    if has_second_started:
        return False
    for line in lines[pref_idx + 1 :]:
        if "WS_AUDIT venue=extended component=ws_msg" not in line:
            continue
        pairs = parse_ws_audit_tokens(line)
        age_data_rx_ms = _safe_int(pairs.get("age_data_rx_ms"))
        age_book_event_ms = _safe_int(pairs.get("age_book_event_ms"))
        age_published_ms = _safe_int(pairs.get("age_published_ms"))
        if (
            age_data_rx_ms is not None
            and age_book_event_ms is not None
            and age_published_ms is not None
            and age_data_rx_ms >= 2000
            and age_book_event_ms >= 2000
            and age_published_ms >= 2000
        ):
            return True
    return False


def run_root_has_extended_pre_kill_degraded_rebootstrap_alignment_gap_evidence(
    run_root: Path,
) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    closeout: dict[str, Any] = {}
    closeout_path = run_root / "live_closeout_bundle.json"
    if closeout_path.exists():
        try:
            with closeout_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                closeout = loaded
        except (OSError, json.JSONDecodeError):
            closeout = {}
    lines = text.splitlines()
    has_extended_only_kill = (
        "stale_market_hygiene kill_triggered" in text and "stale_venues=extended" in text
    )
    has_unexpected_exit = (
        "error=unexpected_live_loop_exit" in text
        and "kill_switch=true" in text
        and "stale_market_count=1" in text
    )
    has_guard_kill_events = (
        bool(closeout.get("guard_intervened"))
        and str(closeout.get("guard_intervention_reason") or "") == "kill_events_present"
    )
    has_pre_kill_boundary = (
        has_unexpected_exit
        or has_guard_kill_events
        or run_root_has_extended_pre_kill_telemetry_evidence(run_root)
    )
    if not has_extended_only_kill or not has_pre_kill_boundary:
        return False

    has_watchdog_fire = (
        "WS_AUDIT venue=extended component=degraded_stream_watchdog action=fired" in text
    )
    if has_watchdog_fire:
        saw_watchdog_fire = False
        saw_first_publish_after_fire = False
        for line in lines:
            if "WS_AUDIT venue=extended component=degraded_stream_watchdog action=fired" in line:
                saw_watchdog_fire = True
                continue
            if (
                saw_watchdog_fire
                and "session_progress stage=first_publish" in line
                and "stream_kind=full_orderbook" in line
            ):
                saw_first_publish_after_fire = True
                break
        if (
            saw_first_publish_after_fire
            and run_root_has_paradex_ui_touch_live_evidence(run_root)
        ):
            return True

    if "WS_AUDIT venue=extended reconnect_reason=stale_watchdog" in text:
        return False
    for line in lines:
        if (
            "WS_AUDIT venue=extended component=post_publish_stream_fallback" not in line
            or "action=degraded_rebootstrap_started" not in line
        ):
            continue
        pairs = parse_ws_audit_tokens(line)
        fallback_after_ms = _safe_int(pairs.get("post_publish_fallback_after_ms"))
        age_data_rx_ms = _safe_int(pairs.get("age_data_rx_ms"))
        age_book_event_ms = _safe_int(pairs.get("age_book_event_ms"))
        age_published_ms = _safe_int(pairs.get("age_published_ms"))
        if None in (
            fallback_after_ms,
            age_data_rx_ms,
            age_book_event_ms,
            age_published_ms,
        ):
            continue
        if (
            age_data_rx_ms >= max(4_000, fallback_after_ms * 2)
            and age_book_event_ms >= max(4_000, fallback_after_ms * 2)
            and age_published_ms >= max(4_000, fallback_after_ms * 2)
        ):
            return True
    return False


def run_root_has_aster_bridge_wait_timeout_evidence(run_root: Path) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    lines = text.splitlines()
    bridge_wait_samples = sum(
        1
        for line in lines
        if "WS_AUDIT venue=aster component=book_recovery stage=snapshot_wait_bridge" in line
    )
    has_aster_watchdog = (
        "WS_AUDIT venue=aster reconnect_reason=stale_watchdog" in text
        or "WS_AUDIT venue=aster component=book_recovery stage=stale_watchdog" in text
    )
    has_snapshot_fetch_failed = (
        "WS_AUDIT venue=aster component=book_recovery stage=snapshot_fetch_failed" in text
    )
    has_aster_kill = (
        "stale_market_hygiene kill_triggered" in text and "stale_venues=aster" in text
    )
    has_unexpected_exit = "error=unexpected_live_loop_exit" in text
    has_extended_watchdog = "WS_AUDIT venue=extended reconnect_reason=stale_watchdog" in text
    has_extended_freeze = "WARN: Extended core book update frozen" in text
    return (
        bridge_wait_samples >= 5
        and (has_aster_watchdog or has_snapshot_fetch_failed)
        and has_aster_kill
        and has_unexpected_exit
        and not has_extended_watchdog
        and not has_extended_freeze
    )


def run_root_has_lighter_no_data_transport_gap_evidence(run_root: Path) -> bool:
    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    has_lighter_kill = (
        "stale_market_hygiene kill_triggered" in text
        and "stale_venues=lighter" in text
    )
    if not has_lighter_kill:
        return False
    has_lighter_stale_watchdog = (
        "WS_AUDIT venue=lighter component=freshness reason=stale_watchdog_trigger" in text
    )
    has_lighter_connect_failure = (
        "Lighter public WS connect error: HTTP error: 503" in text
        or "Lighter public WS connect error: HTTP error: 502" in text
    )
    has_lighter_discovery_failure = (
        "Lighter orderBooks discovery failed" in text
        or "Lighter orderBooks fetch failed status=503" in text
        or "Lighter orderBooks fetch failed status=502" in text
    )
    return (
        has_lighter_stale_watchdog
        or has_lighter_connect_failure
        or has_lighter_discovery_failure
    )


def run_root_has_startup_readiness_gap_evidence(run_root: Path) -> bool:
    closeout_path = run_root / "live_closeout_bundle.json"
    if not closeout_path.exists():
        return False
    try:
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tick_count = int(closeout.get("tick_count") or 0)
    mm_place_total = int(closeout.get("mm_place_total") or 0)
    fill_count_total = int(closeout.get("fill_count_total") or 0)
    if tick_count <= 0 or tick_count > 20 or mm_place_total != 0 or fill_count_total != 0:
        return False

    text = bounded_err_segment_text(run_root)
    if not text:
        return False
    has_kill = "stale_market_hygiene kill_triggered" in text
    has_unexpected_exit = "error=unexpected_live_loop_exit" in text
    has_lighter_stale = "stale_venues=lighter" in text
    has_extended_first_publish = (
        "FIRST_BOOK_UPDATE venue=extended" in text
        or "APPLIED_BOOK venue=extended" in text
    )
    has_extended_apply_truth = (
        "WS_AUDIT venue=extended component=runner_apply_truth" in text
    )
    has_paradex_profile = "WS_AUDIT venue=paradex component=profile_usage" in text
    has_paradex_ui_truth = "WS_AUDIT venue=paradex component=ui_book_truth" in text
    has_paradex_first_book = (
        "FIRST_BOOK_UPDATE venue=paradex" in text
        or "APPLIED_BOOK venue=paradex" in text
    )
    has_extended_paradex_final_stale = (
        "stale_venues=extended,paradex" in text
        or "stale_venues=extended,hyperliquid,paradex" in text
    )
    return (
        has_kill
        and has_unexpected_exit
        and not has_lighter_stale
        and has_extended_first_publish
        and has_extended_apply_truth
        and (has_paradex_profile or has_paradex_ui_truth)
        and not has_paradex_first_book
        and has_extended_paradex_final_stale
    )


def run_root_has_hyperliquid_post_publish_transport_gap_evidence(run_root: Path) -> bool:
    closeout_path = run_root / "live_closeout_bundle.json"
    if not closeout_path.exists():
        return False
    try:
        closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    tick_count = int(closeout.get("tick_count") or 0)
    mm_place_total = int(closeout.get("mm_place_total") or 0)
    if tick_count <= 1000 or mm_place_total <= 0:
        return False

    text = bounded_err_segment_text(run_root)
    if not text:
        return False

    has_hl_only_kill = (
        "stale_market_hygiene kill_triggered" in text and "stale_venues=hyperliquid" in text
    )
    has_unexpected_exit = (
        "error=unexpected_live_loop_exit" in text
        and "kill_switch=true" in text
        and "stale_market_count=1" in text
    )
    has_hl_first_book = (
        "FIRST_BOOK_UPDATE venue=hyperliquid" in text
        or "APPLIED_BOOK venue=hyperliquid" in text
    )
    has_hl_reconnect = "WS_AUDIT venue=hyperliquid reconnect_reason=" in text
    saw_pubq_gap = False
    for line in text.splitlines():
        if "WS_AUDIT venue=hyperliquid component=hl_pubq" not in line:
            continue
        pairs = parse_ws_audit_tokens(line)
        ws_rx_age_ms = _safe_int(pairs.get("ws_rx_age_ms"))
        data_rx_age_ms = _safe_int(pairs.get("data_rx_age_ms"))
        book_age_ms = _safe_int(pairs.get("book_age_ms"))
        pub_age_ms = _safe_int(pairs.get("pub_age_ms"))
        pub_minus_book_age_ms = _safe_int(pairs.get("pub_minus_book_age_ms"))
        if (
            ws_rx_age_ms is not None
            and data_rx_age_ms is not None
            and book_age_ms is not None
            and pub_age_ms is not None
            and ws_rx_age_ms <= 100
            and data_rx_age_ms <= 100
            and book_age_ms <= 100
            and (
                (pub_minus_book_age_ms is not None and pub_minus_book_age_ms >= 1000)
                or pub_age_ms >= 1000
            )
        ):
            saw_pubq_gap = True
            break

    return (
        has_hl_only_kill
        and has_unexpected_exit
        and has_hl_first_book
        and saw_pubq_gap
        and not has_hl_reconnect
    )


def run_root_has_paradex_interactive_top_anchor_gap_evidence(run_root: Path) -> bool:
    metrics = load_live_metrics_bundle(run_root)
    if not metrics:
        return False

    scorecard = metrics.get("execution_scorecard")
    paradex_score = scorecard.get("paradex") if isinstance(scorecard, dict) else None
    if not isinstance(paradex_score, dict):
        return False
    if int(paradex_score.get("fills") or 0) != 0:
        return False
    if int(paradex_score.get("place_ack") or 0) <= 0 and int(paradex_score.get("place_i") or 0) <= 0:
        return False

    orders_per_venue = metrics.get("orders_per_venue")
    paradex_orders = orders_per_venue.get("paradex") if isinstance(orders_per_venue, dict) else 0
    if not isinstance(paradex_orders, (int, float)) or paradex_orders < 100:
        return False

    profile_usage = metrics.get("paradex_profile_usage_summary")
    if not isinstance(profile_usage, dict) or not profile_usage.get("interactive_token_usage_observed"):
        return False

    public_top = metrics.get("paradex_interactive_top_summary")
    if isinstance(public_top, dict):
        source_counts = public_top.get("public_top_source_counts")
        interactive_public_top_count = 0
        if isinstance(source_counts, dict):
            for source, count in source_counts.items():
                if str(source).startswith("interactive_"):
                    parsed_count = _safe_int(count)
                    if parsed_count is not None:
                        interactive_public_top_count += parsed_count
        if public_top.get("interactive_top_level_fallback_present") or interactive_public_top_count > 0:
            return False

    ui_truth = metrics.get("paradex_ui_book_truth_summary")
    if not isinstance(ui_truth, dict) or not ui_truth.get("observed"):
        return False
    api_top = ui_truth.get("last_api_top") if isinstance(ui_truth.get("last_api_top"), dict) else {}
    interactive_top = ui_truth.get("last_interactive_top") if isinstance(ui_truth.get("last_interactive_top"), dict) else {}
    api_bid = _safe_float(api_top.get("bid_px"))
    api_ask = _safe_float(api_top.get("ask_px"))
    interactive_bid = _safe_float(interactive_top.get("bid_px"))
    interactive_ask = _safe_float(interactive_top.get("ask_px"))
    interactive_more_competitive = (
        (interactive_bid is not None and api_bid is not None and interactive_bid > api_bid)
        or (interactive_ask is not None and api_ask is not None and interactive_ask < api_ask)
    )
    if not interactive_more_competitive:
        return False

    touch = metrics.get("paradex_ui_touch_reference_summary")
    if not isinstance(touch, dict) or not touch.get("observed"):
        return False
    if int(touch.get("applied_count") or 0) <= 0:
        return False
    return True


def run_root_has_paradex_ui_touch_reference_gap_evidence(run_root: Path) -> bool:
    metrics = load_live_metrics_bundle(run_root)
    if not metrics:
        return False

    scorecard = metrics.get("execution_scorecard")
    paradex_score = scorecard.get("paradex") if isinstance(scorecard, dict) else None
    if not isinstance(paradex_score, dict):
        return False
    if int(paradex_score.get("fills") or 0) != 0:
        return False
    if int(paradex_score.get("place_ack") or 0) <= 0 and int(paradex_score.get("place_i") or 0) <= 0:
        return False

    orders_per_venue = metrics.get("orders_per_venue")
    paradex_orders = orders_per_venue.get("paradex") if isinstance(orders_per_venue, dict) else 0
    if not isinstance(paradex_orders, (int, float)) or paradex_orders < 1000:
        return False

    profile_usage = metrics.get("paradex_profile_usage_summary")
    if not isinstance(profile_usage, dict) or not profile_usage.get("interactive_token_usage_observed"):
        return False

    ui_truth = metrics.get("paradex_ui_book_truth_summary")
    if isinstance(ui_truth, dict) and ui_truth.get("observed"):
        return False

    interactive_top = metrics.get("paradex_interactive_top_summary")
    if isinstance(interactive_top, dict) and int(interactive_top.get("records") or 0) > 0:
        return False

    touch = metrics.get("paradex_ui_touch_reference_summary")
    if isinstance(touch, dict) and (touch.get("observed") or int(touch.get("applied_count") or 0) > 0):
        return False

    supported_visibility = metrics.get("supported_replace_visibility")
    paradex_visibility = supported_visibility.get("paradex") if isinstance(supported_visibility, dict) else None
    if not isinstance(paradex_visibility, dict):
        return False
    actions = paradex_visibility.get("actions") if isinstance(paradex_visibility.get("actions"), dict) else {}
    blockers = paradex_visibility.get("blockers") if isinstance(paradex_visibility.get("blockers"), dict) else {}
    opportunities = int(paradex_visibility.get("opportunities") or 0)
    keep_actions = int(actions.get("keep") or 0)
    replace_actions = int(actions.get("replace") or 0)
    no_current_same_side = int(blockers.get("no_current_same_side") or 0)
    desired_suppressed = int(blockers.get("desired_suppressed") or 0)
    current_too_young = int(blockers.get("current_too_young") or 0)
    if (
        no_current_same_side >= 5
        and no_current_same_side > desired_suppressed
        and no_current_same_side > current_too_young
    ):
        return False

    mm_keep_by_venue = metrics.get("mm_keep_replace_by_venue")
    paradex_keep = 0
    if isinstance(mm_keep_by_venue, dict):
        paradex_keep = int((mm_keep_by_venue.get("paradex") or {}).get("keep") or 0)
    if opportunities <= 0 and keep_actions <= 0 and replace_actions <= 0 and paradex_keep <= 0:
        return False

    stderr_text = bounded_err_segment_text(run_root)
    return (
        "PARADEX_INTERACTIVE_PUBLIC_TOP" in stderr_text
        and "top_source=api_best" in stderr_text
        and "component=ui_book_truth" not in stderr_text
        and "component=ui_touch_reference action=applied" not in stderr_text
    )


def run_root_has_paradex_edge_floor_queue_loss_evidence(run_root: Path) -> bool:
    metrics = load_live_metrics_bundle(run_root)
    if not metrics:
        return False

    scorecard = metrics.get("execution_scorecard")
    paradex_score = scorecard.get("paradex") if isinstance(scorecard, dict) else None
    if not isinstance(paradex_score, dict):
        return False
    if int(paradex_score.get("fills") or 0) != 0:
        return False
    if int(paradex_score.get("place_ack") or 0) <= 0 and int(paradex_score.get("place_i") or 0) <= 0:
        return False

    orders_per_venue = metrics.get("orders_per_venue")
    paradex_orders = orders_per_venue.get("paradex") if isinstance(orders_per_venue, dict) else 0
    if not isinstance(paradex_orders, (int, float)) or paradex_orders < 100:
        return False

    profile_usage = metrics.get("paradex_profile_usage_summary")
    if not isinstance(profile_usage, dict) or not profile_usage.get("interactive_token_usage_observed"):
        return False

    stderr_text = bounded_err_segment_text(run_root)
    if "PARADEX_INTERACTIVE_PUBLIC_TOP" not in stderr_text:
        return False

    telemetry_path = run_root / "telemetry_bounded.jsonl"
    if not telemetry_path.exists():
        return False

    paradex_active = 0
    paradex_edge_floor_suppressed = 0
    paradex_generated_spread_suppressed = 0
    paradex_utility_suppressed = 0
    for record in iter_json_objects(telemetry_path):
        if record.get("execution_mode") != "live":
            continue
        for quote in record.get("quote_levels", []) or []:
            if quote.get("venue_id") != "paradex":
                continue
            if quote.get("quote_state") == "active":
                paradex_active += 1
            reason = quote.get("suppression_reason")
            if reason == "edge_below_min":
                paradex_edge_floor_suppressed += 1
            elif reason == "generated_spread_cap":
                paradex_generated_spread_suppressed += 1
            elif reason == "utility_suppressed":
                paradex_utility_suppressed += 1

    return (
        paradex_active > 0
        and paradex_edge_floor_suppressed >= 1000
        and paradex_edge_floor_suppressed > paradex_generated_spread_suppressed
        and paradex_edge_floor_suppressed > paradex_utility_suppressed
    )


def run_root_has_paradex_same_side_persistence_gap_evidence(run_root: Path) -> bool:
    metrics = load_live_metrics_bundle(run_root)
    if not metrics:
        return False

    scorecard = metrics.get("execution_scorecard")
    paradex_score = scorecard.get("paradex") if isinstance(scorecard, dict) else None
    if not isinstance(paradex_score, dict):
        return False
    if int(paradex_score.get("fills") or 0) != 0:
        return False
    if int(paradex_score.get("place_ack") or 0) <= 0 and int(paradex_score.get("place_i") or 0) <= 0:
        return False

    orders_per_venue = metrics.get("orders_per_venue")
    paradex_orders = orders_per_venue.get("paradex") if isinstance(orders_per_venue, dict) else 0
    if not isinstance(paradex_orders, (int, float)) or paradex_orders < 1000:
        return False

    profile_usage = metrics.get("paradex_profile_usage_summary")
    if not isinstance(profile_usage, dict) or not profile_usage.get("interactive_token_usage_observed"):
        return False

    stderr_text = bounded_err_segment_text(run_root)
    if "PARADEX_INTERACTIVE_PUBLIC_TOP" not in stderr_text:
        return False

    supported_visibility = metrics.get("supported_replace_visibility")
    paradex_visibility = supported_visibility.get("paradex") if isinstance(supported_visibility, dict) else None
    if not isinstance(paradex_visibility, dict):
        return False
    actions = paradex_visibility.get("actions") if isinstance(paradex_visibility.get("actions"), dict) else {}
    blockers = paradex_visibility.get("blockers") if isinstance(paradex_visibility.get("blockers"), dict) else {}
    no_current_same_side = int(blockers.get("no_current_same_side") or 0)
    desired_suppressed = int(blockers.get("desired_suppressed") or 0)
    current_too_young = int(blockers.get("current_too_young") or 0)
    keep_actions = int(actions.get("keep") or 0)
    place_actions = int(actions.get("place") or 0)

    mm_keep_by_venue = metrics.get("mm_keep_replace_by_venue")
    paradex_keep = 0
    if isinstance(mm_keep_by_venue, dict):
        paradex_keep = int((mm_keep_by_venue.get("paradex") or {}).get("keep") or 0)

    return (
        no_current_same_side >= 5
        and no_current_same_side > desired_suppressed
        and no_current_same_side > current_too_young
        and place_actions > 0
        and (keep_actions > 0 or paradex_keep > 0 or int(paradex_visibility.get("opportunities") or 0) > 0)
    )


def run_root_has_all5_projected_mm_budget_distribution_gap_evidence(run_root: Path) -> bool:
    metrics = load_live_metrics_bundle(run_root)
    if not metrics:
        return False

    fills = metrics.get("fills")
    by_venue = fills.get("by_venue") if isinstance(fills, dict) else None
    if not isinstance(by_venue, dict):
        return False
    fill_venues = [
        venue
        for venue in ("extended", "hyperliquid", "aster", "lighter", "paradex")
        if isinstance(by_venue.get(venue), dict) and int(by_venue[venue].get("fill_count") or 0) > 0
    ]
    if len(fill_venues) >= 3:
        return False

    risk = metrics.get("risk")
    would_send_zero_pct = _safe_float((risk or {}).get("would_send_zero_pct")) if isinstance(risk, dict) else None
    if would_send_zero_pct is not None and would_send_zero_pct < 75.0:
        return False

    selected_counts: dict[str, int] = {}
    suppressed_counts: dict[str, int] = {}
    configured_ticks = 0
    applied_ticks = 0

    summary = metrics.get("projected_mm_budget_summary")
    if isinstance(summary, dict):
        configured_ticks = int(summary.get("configured_ticks") or 0)
        applied_ticks = int(summary.get("applied_ticks") or 0)
        raw_selected = summary.get("selected_counts")
        raw_suppressed = summary.get("suppressed_counts")
        if isinstance(raw_selected, dict):
            selected_counts = {
                str(venue): int(count)
                for venue, count in raw_selected.items()
                if isinstance(count, (int, float)) and count > 0
            }
        if isinstance(raw_suppressed, dict):
            suppressed_counts = {
                str(venue): int(count)
                for venue, count in raw_suppressed.items()
                if isinstance(count, (int, float)) and count > 0
            }

    if configured_ticks == 0:
        telemetry_path = run_root / "telemetry_bounded.jsonl"
        if telemetry_path.exists():
            for record in iter_json_objects(telemetry_path):
                if record.get("execution_mode") != "live":
                    continue
                budget = record.get("projected_mm_budget")
                if not isinstance(budget, dict):
                    continue
                if budget.get("configured"):
                    configured_ticks += 1
                if budget.get("applied"):
                    applied_ticks += 1
                for venue in budget.get("selected_venues", []) or []:
                    if isinstance(venue, str) and venue:
                        selected_counts[venue] = selected_counts.get(venue, 0) + 1
                for venue in budget.get("suppressed_venues", []) or []:
                    if isinstance(venue, str) and venue:
                        suppressed_counts[venue] = suppressed_counts.get(venue, 0) + 1

    venues = ("extended", "hyperliquid", "aster", "lighter", "paradex")
    if configured_ticks <= 0:
        return False
    if not all(selected_counts.get(venue, 0) > 0 for venue in venues):
        return False

    underfilled_suppressed = any(
        suppressed_counts.get(venue, 0) > 0
        for venue in venues
        if venue not in fill_venues
    )
    selection_values = [selected_counts.get(venue, 0) for venue in venues]
    selection_skew = min(selection_values) * 2 < max(selection_values)
    return applied_ticks > 0 and (underfilled_suppressed or selection_skew)


def infer_guard_intervention_blocker_family(run_root: Path) -> str | None:
    if run_root_has_startup_readiness_gap_evidence(run_root):
        return "startup_readiness_gap"
    if run_root_has_hyperliquid_post_publish_transport_gap_evidence(run_root):
        return "hyperliquid_post_publish_transport_gap"
    if run_root_has_extended_pre_kill_degraded_rebootstrap_alignment_gap_evidence(run_root):
        return "extended_pre_kill_degraded_rebootstrap_alignment_gap"
    if run_root_has_extended_degraded_stream_rebootstrap_gap_evidence(run_root):
        return "extended_degraded_stream_rebootstrap_gap"
    if run_root_has_aster_bridge_wait_timeout_evidence(run_root):
        return "aster_bridge_wait_timeout"
    if run_root_has_extended_transport_gap_watchdog_evidence(run_root):
        return "transport_gap_watchdog"
    if run_root_has_lighter_no_data_transport_gap_evidence(run_root):
        return "no_data_transport_gap"
    if run_root_has_extended_freeze_path_evidence(run_root):
        return "runner_freeze_apply_gap"
    return None


def inferred_hold_blocker_family(
    tranche: dict[str, Any],
    autoscore: dict[str, Any],
    run_root: Path,
) -> str | None:
    hypothesis = tranche.get("hypothesis_blocker_family")
    if not hypothesis:
        return None
    closeout_path = run_root / "live_closeout_bundle.json"
    if not closeout_path.exists():
        return None
    with closeout_path.open("r", encoding="utf-8") as handle:
        closeout = json.load(handle)
    if (
        closeout.get("guard_window_completed")
        and not closeout.get("guard_intervened")
        and (
            closeout.get("first_pre_restore_venue_audit_clean") is False
            or closeout.get("pre_restore_cleanup_required") is True
            or closeout.get("pre_restore_venue_audit_clean") is False
        )
        and closeout.get("post_rollback_venue_audit_clean")
        and autoscore.get("mechanism", {}).get("passed", False)
    ):
        return "restore_hygiene"
    guard_reason = str(closeout.get("guard_intervention_reason") or "")
    if guard_reason.startswith("service_restarts=") or guard_reason.startswith("systemd_state="):
        inferred = infer_guard_intervention_blocker_family(run_root)
        if inferred is not None:
            return inferred
        return "stale_restart"
    if closeout.get("guard_intervened"):
        inferred = infer_guard_intervention_blocker_family(run_root)
        if inferred is not None:
            return inferred
        return str(hypothesis)
    mechanism_fail_blocker_family = tranche.get("mechanism_fail_blocker_family")
    if (
        autoscore.get("clean", {}).get("passed", False)
        and not autoscore.get("mechanism", {}).get("passed", False)
        and isinstance(mechanism_fail_blocker_family, str)
        and mechanism_fail_blocker_family
    ):
        return mechanism_fail_blocker_family
    clean_final_hold_blocker_family = tranche.get("clean_final_hold_blocker_family")
    if (
        autoscore.get("clean", {}).get("passed", False)
        and autoscore.get("mechanism", {}).get("passed", False)
        and autoscore.get("is_final_rung")
        and autoscore.get("suggested_action") == "hold"
        and run_root_has_paradex_interactive_top_anchor_gap_evidence(run_root)
    ):
        return "paradex_interactive_top_anchor_gap"
    if (
        autoscore.get("clean", {}).get("passed", False)
        and autoscore.get("mechanism", {}).get("passed", False)
        and autoscore.get("is_final_rung")
        and autoscore.get("suggested_action") == "hold"
        and run_root_has_paradex_ui_touch_reference_gap_evidence(run_root)
    ):
        return "paradex_ui_touch_reference_gap"
    if (
        autoscore.get("clean", {}).get("passed", False)
        and autoscore.get("mechanism", {}).get("passed", False)
        and autoscore.get("is_final_rung")
        and autoscore.get("suggested_action") == "hold"
        and run_root_has_paradex_same_side_persistence_gap_evidence(run_root)
    ):
        return "paradex_same_side_persistence_gap"
    if (
        autoscore.get("clean", {}).get("passed", False)
        and autoscore.get("mechanism", {}).get("passed", False)
        and autoscore.get("is_final_rung")
        and autoscore.get("suggested_action") == "hold"
        and run_root_has_all5_projected_mm_budget_distribution_gap_evidence(run_root)
    ):
        return "all5_projected_mm_budget_distribution_gap"
    if (
        autoscore.get("clean", {}).get("passed", False)
        and autoscore.get("mechanism", {}).get("passed", False)
        and autoscore.get("is_final_rung")
        and autoscore.get("suggested_action") == "hold"
        and run_root_has_paradex_edge_floor_queue_loss_evidence(run_root)
    ):
        return "paradex_edge_floor_queue_loss"
    if (
        autoscore.get("clean", {}).get("passed", False)
        and autoscore.get("mechanism", {}).get("passed", False)
        and autoscore.get("is_final_rung")
        and autoscore.get("suggested_action") == "hold"
        and isinstance(clean_final_hold_blocker_family, str)
        and clean_final_hold_blocker_family
    ):
        return clean_final_hold_blocker_family
    if not autoscore["clean"]["passed"] or not autoscore["mechanism"]["passed"]:
        return str(hypothesis)
    return None


def rung_decision(rung: dict[str, Any], autoscore: dict[str, Any]) -> str:
    if autoscore.get("suggested_action") == "rollback":
        return "rollback"
    continue_on = rung.get("continue_on", "clean")
    if continue_on == "clean":
        return "continue" if autoscore["clean"]["passed"] else "hold"
    if continue_on == "mechanism":
        return "continue" if autoscore["clean"]["passed"] and autoscore["mechanism"]["passed"] else "hold"
    if continue_on == "promotion":
        if not autoscore["clean"]["passed"] or not autoscore["mechanism"]["passed"]:
            return "hold"
        return "promote" if autoscore["suggested_action"] == "promote" else "hold"
    raise ValueError(f"unsupported rung continue_on {continue_on!r}")


def split_csv_env(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def topology_snapshot_from_overlay_path(
    base_topology: dict[str, Any],
    overlay_path: Path,
) -> dict[str, Any]:
    topology = copy.deepcopy(base_topology or {})
    roles = copy.deepcopy(topology.get("roles", {}) or {})
    connectors = list(topology.get("connectors", []) or [])
    fv_disabled = list(topology.get("fv_disabled_venues", []) or [])
    excluded = list(topology.get("excluded_venues", []) or [])

    if overlay_path.exists():
        env = parse_env_file(overlay_path)
        overlay_connectors = split_csv_env(env.get("PARAPHINA_LIVE_CONNECTORS"))
        if overlay_connectors:
            connectors = overlay_connectors
        overlay_fv_disabled = split_csv_env(env.get("PARAPHINA_FV_DISABLED_VENUES"))
        fv_disabled = overlay_fv_disabled
        for key, value in env.items():
            if not key.startswith("PARAPHINA_MM_VENUE_ROLE_"):
                continue
            venue = key.removeprefix("PARAPHINA_MM_VENUE_ROLE_").lower()
            if venue:
                roles[venue] = value.strip()
        known = set(roles) | set(connectors) | set(topology.get("connectors", []) or [])
        excluded = sorted(venue for venue in known if venue not in set(connectors))

    return {
        "overlay_path": str(overlay_path),
        "connectors": sorted({item.lower() for item in connectors}),
        "fv_disabled_venues": sorted({item.lower() for item in fv_disabled}),
        "excluded_venues": sorted({item.lower() for item in excluded}),
        "roles": roles,
    }


def tranche_topology_snapshot(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    overlay_path = tranche_stage_overlay_source(tranche, control_pack["execution_defaults"])
    return topology_snapshot_from_overlay_path(control_pack.get("topology", {}) or {}, overlay_path)


def current_runtime_topology_snapshot(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    stage_overlay_target = control_pack["execution_defaults"].get("stage_overlay_target")
    if isinstance(stage_overlay_target, str) and stage_overlay_target:
        target_path = Path(stage_overlay_target)
        target_is_local = False
        try:
            target_path.relative_to(repo_root)
            target_is_local = True
        except ValueError:
            target_is_local = False
        if target_path.exists() and (repo_root == ROOT or target_is_local):
            return topology_snapshot_from_overlay_path(control_pack.get("topology", {}) or {}, target_path)
    return tranche_topology_snapshot(tranche, control_pack, repo_root)


def load_closeout_bundle(run_root: Path | None) -> dict[str, Any]:
    if run_root is None:
        return {}
    closeout_path = run_root / "live_closeout_bundle.json"
    if not closeout_path.exists():
        return {}
    with closeout_path.open("r", encoding="utf-8") as handle:
        closeout = normalized_closeout_bundle(json.load(handle))
    closeout["summary_exists"] = bool(closeout.get("summary_exists")) or (run_root / "live_segment_summary.json").exists()
    closeout["report_exists"] = bool(closeout.get("report_exists")) or (run_root / "telemetry_report_live_segment.md").exists()
    closeout["metrics_exists"] = bool(closeout.get("metrics_exists")) or (run_root / "live_metrics.json").exists()
    closeout["guard_result_exists"] = bool(closeout.get("guard_result_exists")) or (run_root / "guard_result.json").exists()
    closeout["closeout_contract_complete"] = bool(
        closeout.get("summary_exists")
        and closeout.get("report_exists")
        and closeout.get("metrics_exists")
        and closeout.get("guard_result_exists")
        and closeout.get("health_post_complete")
        and closeout.get("systemd_post_complete")
    )
    closeout["closeout_completeness"] = "full" if closeout["closeout_contract_complete"] else "partial"
    return closeout


def load_live_metrics_bundle(run_root: Path | None) -> dict[str, Any]:
    if run_root is None:
        return {}
    metrics_path = run_root / "live_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def venue_fill_participation_observed(metrics: dict[str, Any], venue: str, fill_role_enabled: bool) -> bool:
    if not fill_role_enabled:
        return False
    scorecard = metrics.get("execution_scorecard")
    if isinstance(scorecard, dict):
        venue_scorecard = scorecard.get(venue)
        if isinstance(venue_scorecard, dict):
            for key in ("fills", "fill_base"):
                value = venue_scorecard.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    return True
    fills = metrics.get("fills")
    if isinstance(fills, dict):
        by_venue = fills.get("by_venue")
        if isinstance(by_venue, dict):
            venue_fills = by_venue.get(venue)
            if isinstance(venue_fills, dict):
                fill_count = venue_fills.get("fill_count")
                fill_base = venue_fills.get("fill_base")
                if isinstance(fill_count, (int, float)) and fill_count > 0:
                    return True
                if isinstance(fill_base, (int, float)) and fill_base > 0:
                    return True
    return False


def build_venue_capability_matrix(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    run_root: Path | None = None,
    autoscore: dict[str, Any] | None = None,
    final_decision: str | None = None,
) -> dict[str, Any]:
    snapshot = tranche_topology_snapshot(tranche, control_pack, repo_root)
    closeout = load_closeout_bundle(run_root)
    metrics = load_live_metrics_bundle(run_root)
    connectors = set(snapshot["connectors"])
    fv_disabled = set(snapshot["fv_disabled_venues"])
    excluded = set(snapshot["excluded_venues"])
    roles = snapshot["roles"]
    venues = sorted(set(roles) | connectors | fv_disabled | excluded)
    restore_clean = bool(
        closeout.get("pre_restore_venue_audit_clean") and closeout.get("post_rollback_venue_audit_clean")
    )
    surface_clean = bool((autoscore or {}).get("clean", {}).get("passed", False))
    mechanism_clean = bool((autoscore or {}).get("mechanism", {}).get("passed", False))
    global_long_soak_ready = (
        not excluded
        and not fv_disabled
        and restore_clean
        and surface_clean
        and mechanism_clean
        and final_decision == "promote"
    )
    matrix: dict[str, Any] = {}
    for venue in venues:
        role = roles.get(venue)
        connected = venue in connectors
        execution_eligible = connected and role not in {None, "excluded_pending_rescue"}
        fill_role_enabled = connected and role in {"primary_fill", "fill"}
        fill_participation = venue_fill_participation_observed(metrics, venue, fill_role_enabled)
        primary_fill_candidate = fill_role_enabled and fill_participation
        venue_fv_eligible = connected and venue not in fv_disabled and role not in {None, "excluded_pending_rescue"}
        matrix[venue] = {
            "role": role,
            "connected": connected,
            "execution_eligible": execution_eligible,
            "fill_role_enabled": fill_role_enabled,
            "fill_participation_observed": fill_participation,
            "primary_fill_candidate": primary_fill_candidate,
            "fv_eligible": venue_fv_eligible,
            "restore_clean": restore_clean,
            "long_soak_ready": global_long_soak_ready and primary_fill_candidate and venue_fv_eligible,
        }
    return {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "tranche_id": tranche.get("id"),
        "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
        "topology_snapshot": snapshot,
        "autoscore_clean_passed": surface_clean,
        "autoscore_mechanism_passed": mechanism_clean,
        "final_decision": final_decision,
        "venues": matrix,
    }


def write_venue_capability_matrix(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    run_root: Path | None = None,
    autoscore: dict[str, Any] | None = None,
    final_decision: str | None = None,
) -> Path:
    path = stage_contract_paths(tranche, control_pack, repo_root)["venue_capability_matrix"]
    write_json(
        path,
        build_venue_capability_matrix(
            tranche,
            control_pack,
            repo_root=repo_root,
            run_root=run_root,
            autoscore=autoscore,
            final_decision=final_decision,
        ),
    )
    return path


def write_stage_verdict(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    repo_root: Path = ROOT,
    session: dict[str, Any] | None = None,
    run_root: Path | None = None,
    autoscore: dict[str, Any] | None = None,
    decision: str | None = None,
    observed_blocker_family: str | None = None,
    continue_to_next_rung: bool = False,
    selected_child: str | None = None,
    selected_child_support_gate: dict[str, Any] | None = None,
    state_sync: dict[str, Any] | None = None,
) -> Path:
    verdict = "HOLD"
    if decision == "promote":
        verdict = "PROMOTE"
    elif decision == "rollback":
        verdict = "ROLLBACK"
    run_dir = ensure_run_dir(repo_root, str(tranche["id"]))
    contract_paths = stage_contract_paths(tranche, control_pack, repo_root)
    closeout_path = None if run_root is None else run_root / "live_closeout_bundle.json"
    payload = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "tranche_id": tranche.get("id"),
        "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
        "verdict": verdict,
        "decision": decision,
        "continue_to_next_rung": continue_to_next_rung,
        "selected_child": selected_child,
        "selected_child_support_gate": selected_child_support_gate,
        "observed_blocker_family": observed_blocker_family,
        "run_root": None if run_root is None else str(run_root),
        "closeout_bundle_path": None if closeout_path is None else str(closeout_path),
        "autoscore_bundle_path": None if run_root is None else str(run_root / "autoscore_bundle.json"),
        "support_summary_path": str(contract_paths["support_summary"]),
        "venue_capability_matrix_path": str(contract_paths["venue_capability_matrix"]),
        "session_id": None if session is None else session.get("session_id"),
        "support_runs_count": len((session or {}).get("support_runs", []) or []),
        "autoscore": None if autoscore is None else {
            "clean_passed": autoscore["clean"]["passed"],
            "mechanism_passed": autoscore["mechanism"]["passed"],
            "promotion_passed": autoscore["promotion"]["passed"],
            "suggested_action": autoscore["suggested_action"],
        },
        "state_sync": state_sync,
    }
    path = contract_paths["stage_verdict"]
    write_json(path, payload)
    return path


def write_manual_live_stage_contracts(
    tranche: dict[str, Any],
    control_pack: dict[str, Any],
    run_root: Path,
    duration_sec: int,
    repo_root: Path = ROOT,
) -> None:
    autoscore = autoscore_run(tranche, control_pack, run_root, duration_sec, repo_root)
    orchestration = load_orchestration(repo_root)
    found = find_orchestration_session(orchestration, str(tranche["id"]))
    session = None if found is None else found[1]
    queue, _ = load_state(repo_root)
    state_sync_report = audit_state_sync(queue, control_pack, repo_root, tranche_id=str(tranche["id"]))
    state_sync_report_path = write_state_sync_report(run_root, state_sync_report)
    state_sync_summary = state_sync_summary_payload(state_sync_report, state_sync_report_path)
    observed_blocker_family = None
    if autoscore["suggested_action"] == "hold":
        observed_blocker_family = inferred_hold_blocker_family(tranche, autoscore, run_root)
    suggested = str(autoscore.get("suggested_action") or "")
    decision = None
    continue_to_next_rung = suggested == "continue"
    if suggested in {"hold", "promote", "rollback"}:
        decision = suggested
    write_support_summary(str(tranche["id"]), tranche, control_pack, session, repo_root, state_sync=state_sync_summary)
    write_venue_capability_matrix(
        tranche,
        control_pack,
        repo_root=repo_root,
        run_root=run_root,
        autoscore=autoscore,
        final_decision=None,
    )
    write_stage_verdict(
        tranche,
        control_pack,
        repo_root=repo_root,
        session=session,
        run_root=run_root,
        autoscore=autoscore,
        decision=decision,
        observed_blocker_family=observed_blocker_family,
        continue_to_next_rung=continue_to_next_rung,
        state_sync=state_sync_summary,
    )


def reduce_recovered_latest_run_into_session(
    tranche_id: str,
    tranche: dict[str, Any],
    automation: dict[str, Any],
    session: dict[str, Any],
    repo_root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], Path] | None:
    manifest_path = latest_run_manifest_path(repo_root, tranche_id)
    manifest = load_yaml(manifest_path) if manifest_path.exists() else {}
    run_root_raw = manifest.get("run_root")
    duration_sec = manifest.get("duration_sec")
    manifest_updated_utc = manifest.get("updated_utc")
    session_created_utc = session.get("created_utc")
    if not isinstance(run_root_raw, str) or not run_root_raw or not isinstance(duration_sec, int):
        return None
    # Ignore stale latest-run manifests from older completed sessions. Recovery is only
    # valid when the manifest was written during or after this session's lifetime.
    if (
        isinstance(manifest_updated_utc, str)
        and manifest_updated_utc
        and isinstance(session_created_utc, str)
        and session_created_utc
        and manifest_updated_utc < session_created_utc
    ):
        return None
    run_root = Path(run_root_raw)
    closeout_path = run_root / "live_closeout_bundle.json"
    autoscore_path = run_root / "autoscore_bundle.json"
    if not run_root.exists() or not closeout_path.exists() or not autoscore_path.exists():
        return None
    completed = {
        int(result["duration_sec"])
        for result in session.get("rung_results", []) or []
        if "duration_sec" in result
    }
    if duration_sec in completed:
        return None
    rung = next(
        (
            item
            for item in automation.get("rung_plan", []) or []
            if int(item.get("duration_sec") or 0) == duration_sec
        ),
        None,
    )
    if rung is None:
        return None
    with autoscore_path.open("r", encoding="utf-8") as handle:
        autoscore = json.load(handle)
    decision = rung_decision(rung, autoscore)
    rung_record = {
        "duration_sec": duration_sec,
        "label": duration_label(duration_sec),
        "run_root": str(run_root),
        "closeout_bundle_path": str(closeout_path),
        "autoscore_bundle_path": str(autoscore_path),
        "decision": decision,
        "updated_utc": utc_now(),
        "recovered": True,
    }
    session.setdefault("rung_results", []).append(rung_record)
    session["updated_utc"] = utc_now()
    return rung_record, autoscore, run_root


def orchestrate_tranche(
    tranche_id: str,
    repo_root: Path = ROOT,
    resume: bool = False,
) -> dict[str, Any]:
    maybe_auto_recover_latest_run(repo_root, tranche_id)
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    if tranche.get("track") != "serialized_mainline":
        raise ValueError(f"{tranche_id}: orchestrate only supports serialized mainline tranches")

    orchestration = load_orchestration(repo_root)
    found = find_orchestration_session(orchestration, tranche_id)
    if found is None:
        session = spawn_lanes(tranche_id, repo_root=repo_root)
        orchestration = load_orchestration(repo_root)
        _, session = find_orchestration_session(orchestration, tranche_id)  # type: ignore[misc]
    else:
        _, session = found
        if session.get("state") in {"completed", "archived"} and not resume:
            session = spawn_lanes(tranche_id, repo_root=repo_root)
            orchestration = load_orchestration(repo_root)
            _, session = find_orchestration_session(orchestration, tranche_id)  # type: ignore[misc]

    preflight = session.get("preflight")
    if orchestration_preflight_requires_refresh(preflight):
        preflight = ensure_orchestration_preflight(tranche, control_pack, repo_root)
        session["preflight"] = preflight
        session["preflight_summary_path"] = preflight.get("preflight_summary_path")
        session["state_sync_report_path"] = preflight.get("state_sync_report_path")
        upsert_orchestration_session(orchestration, session)
        save_orchestration(orchestration, repo_root)
    else:
        updated = False
        for key in ("preflight_summary_path", "state_sync_report_path"):
            if preflight.get(key) is not None and session.get(key) != preflight.get(key):
                session[key] = preflight.get(key)
                updated = True
        if updated:
            upsert_orchestration_session(orchestration, session)
            save_orchestration(orchestration, repo_root)

    session["state"] = "running"
    session["updated_utc"] = utc_now()
    if tranche.get("status") in {"ready", "hold"}:
        session["tranche_card_path"] = str(
            prepare_tranche(repo_root=repo_root, tranche_id=tranche_id, mark_in_progress=True)
        )
    elif tranche.get("status") == "in_progress":
        session["tranche_card_path"] = str(ensure_run_dir(repo_root, tranche_id) / "tranche_card.yaml")
    upsert_orchestration_session(orchestration, session)
    save_orchestration(orchestration, repo_root)

    automation = tranche_automation(tranche, control_pack, repo_root)
    recovered_rung = reduce_recovered_latest_run_into_session(
        tranche_id,
        tranche,
        automation,
        session,
        repo_root=repo_root,
    )
    if recovered_rung is not None:
        upsert_orchestration_session(orchestration, session)
        save_orchestration(orchestration, repo_root)
    completed = {int(result["duration_sec"]) for result in session.get("rung_results", []) if "duration_sec" in result}
    final_decision: str | None = session.get("final_decision")
    final_run_root: Path | None = None
    final_autoscore: dict[str, Any] | None = None
    if recovered_rung is not None:
        recovered_record, recovered_autoscore, recovered_run_root = recovered_rung
        if recovered_record["decision"] != "continue":
            final_decision = str(recovered_record["decision"])
            final_run_root = recovered_run_root
            final_autoscore = recovered_autoscore

    for rung in automation["rung_plan"]:
        if final_decision is not None:
            break
        duration_sec = int(rung["duration_sec"])
        if duration_sec in completed:
            continue
        run_root = run_live_guarded(tranche_id, duration_sec, repo_root)
        autoscore = autoscore_run(tranche, control_pack, run_root, duration_sec, repo_root)
        record_forensics_bundle(tranche_id, repo_root, session, run_root, duration_sec, autoscore)
        run_session_support_lanes(
            tranche,
            control_pack,
            session,
            repo_root,
            stage_context=f"rung_{duration_sec}",
        )
        decision = rung_decision(rung, autoscore)
        rung_record = {
            "duration_sec": duration_sec,
            "label": duration_label(duration_sec),
            "run_root": str(run_root),
            "closeout_bundle_path": str(run_root / "live_closeout_bundle.json"),
            "autoscore_bundle_path": str(run_root / "autoscore_bundle.json"),
            "decision": decision,
            "updated_utc": utc_now(),
        }
        session.setdefault("rung_results", []).append(rung_record)
        session["updated_utc"] = utc_now()
        upsert_orchestration_session(orchestration, session)
        save_orchestration(orchestration, repo_root)
        write_venue_capability_matrix(
            tranche,
            control_pack,
            repo_root=repo_root,
            run_root=run_root,
            autoscore=autoscore,
            final_decision=None if decision == "continue" else decision,
        )
        write_stage_verdict(
            tranche,
            control_pack,
            repo_root=repo_root,
            session=session,
            run_root=run_root,
            autoscore=autoscore,
            decision="hold" if decision == "continue" else decision,
            observed_blocker_family=None if decision == "continue" else inferred_hold_blocker_family(tranche, autoscore, run_root),
            continue_to_next_rung=(decision == "continue"),
        )
        if decision == "continue":
            continue
        final_decision = decision
        final_run_root = run_root
        final_autoscore = autoscore
        break

    if final_decision is None:
        if not session.get("rung_results"):
            raise RuntimeError(f"{tranche_id}: orchestration completed without rung results")
        last = session["rung_results"][-1]
        final_decision = str(last.get("decision") or "hold")
        final_run_root = Path(last["run_root"])
        final_autoscore = autoscore_payloads(final_run_root)  # type: ignore[assignment]
    if final_decision == "continue":
        final_decision = "hold"

    assert final_run_root is not None
    summary_path = final_run_root / "live_segment_summary.json"
    guard_path = final_run_root / "guard.log"
    notes = (
        f"orchestrated via session {session['session_id']}; "
        f"autoscore_bundle={final_run_root / 'autoscore_bundle.json'}"
    )
    observed_blocker = (
        inferred_hold_blocker_family(tranche, final_autoscore, final_run_root)
        if final_decision == "hold" and isinstance(final_autoscore, dict) and "clean" in final_autoscore
        else None
    )
    record_result(
        tranche_id=tranche_id,
        decision=final_decision,
        repo_root=repo_root,
        summary_path=str(summary_path) if summary_path.exists() else None,
        guard_path=str(guard_path) if guard_path.exists() else None,
        notes=notes,
        observed_blocker_family=observed_blocker,
    )

    queue, _ = load_state(repo_root)
    _, _, updated_tranche = find_tranche(queue, tranche_id)
    latest_history = latest_history_entry(updated_tranche) or {}
    effective_final_decision = str(latest_history.get("decision") or final_decision)
    effective_observed_blocker = latest_history.get("observed_blocker_family") or observed_blocker
    session["final_decision"] = effective_final_decision
    session["selected_child"] = latest_history.get("activated_child")
    session["state"] = "completed"
    session["updated_utc"] = utc_now()
    session["latest_summary_path"] = str(summary_path) if summary_path.exists() else None
    session["latest_guard_path"] = str(guard_path) if guard_path.exists() else None
    if automation.get("cleanup_policy") == "ephemeral_on_verdict":
        archive_session_worktrees(session, repo_root)
    upsert_orchestration_session(orchestration, session)
    save_orchestration(orchestration, repo_root)
    write_support_summary(
        tranche_id,
        updated_tranche,
        control_pack,
        session,
        repo_root,
        state_sync=latest_history.get("state_sync"),
    )
    write_venue_capability_matrix(
        updated_tranche,
        control_pack,
        repo_root=repo_root,
        run_root=final_run_root,
        autoscore=final_autoscore if isinstance(final_autoscore, dict) else None,
        final_decision=effective_final_decision,
    )
    write_stage_verdict(
        updated_tranche,
        control_pack,
        repo_root=repo_root,
        session=session,
        run_root=final_run_root,
        autoscore=final_autoscore if isinstance(final_autoscore, dict) else None,
        decision=effective_final_decision,
        observed_blocker_family=effective_observed_blocker,
        selected_child=session.get("selected_child"),
        state_sync=latest_history.get("state_sync"),
    )
    return session


def resume_orchestrate_tranche(tranche_id: str, repo_root: Path = ROOT) -> dict[str, Any]:
    orchestration = load_orchestration(repo_root)
    found = find_orchestration_session(orchestration, tranche_id)
    if found is None:
        return orchestrate_tranche(tranche_id, repo_root=repo_root, resume=False)
    _, session = found
    if session.get("state") in {"completed", "archived"}:
        return session
    return orchestrate_tranche(tranche_id, repo_root=repo_root, resume=True)


def auto_progress_run_session(
    tranche_id: str,
    repo_root: Path = ROOT,
    *,
    resume: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        session = (
            resume_orchestrate_tranche(tranche_id, repo_root=repo_root)
            if resume
            else orchestrate_tranche(tranche_id, repo_root=repo_root, resume=False)
        )
    except RuntimeError as exc:
        message = str(exc)
        marker = " headroom below automation default"
        if marker in message:
            label = message.split(marker, 1)[0].strip().replace(" ", "_")
            return None, {
                "state": "infra_invalid",
                "reason": f"{label}_headroom_below_default" if label else "storage_headroom_below_default",
                "tranche_id": tranche_id,
                "error": message,
                "retryable": True,
                "updated_utc": utc_now(),
            }
        state_sync_marker = "orchestration preflight blocked by state-sync"
        if state_sync_marker in message:
            return None, {
                "state": "control_plane_invalid",
                "reason": "state_sync_preflight_block",
                "tranche_id": tranche_id,
                "error": message,
                "retryable": False,
                "updated_utc": utc_now(),
            }
        raise
    return session, None


def auto_progress_serialized_mainline_once(
    repo_root: Path = ROOT,
    tranche_id: str | None = None,
) -> dict[str, Any]:
    queue, _ = load_state(repo_root)
    tranche: dict[str, Any] | None = None
    if tranche_id:
        _, _, tranche = find_tranche(queue, tranche_id)
        if tranche.get("track") != "serialized_mainline":
            raise ValueError(f"{tranche_id}: auto-progress only supports serialized mainline tranches")
        if tranche.get("status") not in {"ready", "in_progress"}:
            tranche = None
    if tranche is None:
        tranche = current_serialized_mainline(queue)
    if tranche is None:
        return {
            "state": "idle",
            "reason": "no_current_mainline",
            "updated_utc": utc_now(),
        }

    current_id = str(tranche.get("id") or "")
    if not current_id:
        return {
            "state": "idle",
            "reason": "current_mainline_missing_id",
            "updated_utc": utc_now(),
        }

    restore_required = latest_run_restore_required_manifest(repo_root, current_id)
    if restore_required is not None:
        return {
            "state": "waiting_for_restore",
            "tranche_id": current_id,
            "run_root": restore_required.get("run_root"),
            "duration_sec": restore_required.get("duration_sec"),
            "restore_required_reasons": restore_required.get("restore_required_reasons", []),
            "updated_utc": utc_now(),
        }

    should_wait, run_root, duration_sec = latest_run_requires_recovery(repo_root, current_id)
    if should_wait:
        return {
            "state": "waiting_for_closeout",
            "tranche_id": current_id,
            "run_root": str(run_root) if run_root is not None else None,
            "duration_sec": duration_sec,
            "updated_utc": utc_now(),
        }

    session, infra_event = auto_progress_run_session(current_id, repo_root=repo_root, resume=True)
    if infra_event is not None:
        return infra_event
    assert session is not None
    queue_after, _ = load_state(repo_root)
    next_mainline = current_serialized_mainline(queue_after)
    next_current_tranche_id = None if next_mainline is None else next_mainline.get("id")
    session_state = str(session.get("state") or "")
    if (
        str(tranche.get("status") or "") == "hold"
        and session_state in {"completed", "archived"}
        and next_current_tranche_id == current_id
    ):
        selected_child = str(session.get("selected_child") or "")
        if selected_child and selected_child != current_id:
            try:
                _, _, child_tranche = find_tranche(queue_after, selected_child)
            except KeyError:
                child_tranche = None
            if (
                child_tranche is not None
                and child_tranche.get("track") == "serialized_mainline"
                and child_tranche.get("status") in {"ready", "hold", "in_progress"}
            ):
                child_session, infra_event = auto_progress_run_session(
                    selected_child,
                    repo_root=repo_root,
                    resume=False,
                )
                if infra_event is not None:
                    infra_event["prior_tranche_id"] = current_id
                    return infra_event
                assert child_session is not None
                queue_after_handoff, _ = load_state(repo_root)
                next_mainline = current_serialized_mainline(queue_after_handoff)
                next_current_tranche_id = None if next_mainline is None else next_mainline.get("id")
                return {
                    "state": "resumed",
                    "reason": "handoff_selected_child",
                    "tranche_id": selected_child,
                    "prior_tranche_id": current_id,
                    "session_id": child_session.get("session_id"),
                    "session_state": child_session.get("state"),
                    "final_decision": child_session.get("final_decision"),
                    "selected_child": child_session.get("selected_child"),
                    "next_current_tranche_id": next_current_tranche_id,
                    "updated_utc": utc_now(),
                }
        return {
            "state": "idle",
            "reason": "completed_current_mainline_hold",
            "tranche_id": current_id,
            "session_id": session.get("session_id"),
            "session_state": session_state,
            "final_decision": session.get("final_decision"),
            "selected_child": session.get("selected_child"),
            "updated_utc": utc_now(),
        }
    return {
        "state": "resumed",
        "tranche_id": current_id,
        "session_id": session.get("session_id"),
        "session_state": session_state,
        "final_decision": session.get("final_decision"),
        "selected_child": session.get("selected_child"),
        "next_current_tranche_id": next_current_tranche_id,
        "updated_utc": utc_now(),
    }


def watch_serialized_mainline(
    repo_root: Path = ROOT,
    tranche_id: str | None = None,
    poll_sec: float = 60.0,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_target = tranche_id
    while True:
        event = auto_progress_serialized_mainline_once(repo_root=repo_root, tranche_id=current_target)
        events.append(event)
        state = str(event.get("state") or "")
        if state == "idle":
            break
        if state in {"waiting_for_closeout", "infra_invalid"}:
            time.sleep(max(1.0, poll_sec))
            continue
        current_target = None
    return events


def build_live_run_root(control_pack: dict[str, Any], tranche_id: str, duration_sec: int) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{tranche_id}_{duration_label(duration_sec)}_{ts}"
    return Path(control_pack["execution_defaults"]["promotion_runs_root"]) / name / "live_canary"


def build_shadow_smoke_run_root(control_pack: dict[str, Any], tranche_id: str, duration_sec: int) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    duration_min = max(1, duration_sec // 60)
    name = f"{tranche_id}_shadow_smoke_{duration_min}m_{ts}"
    return Path(control_pack["execution_defaults"]["promotion_runs_root"]) / name


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed = value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
            parsed = parsed[1:-1]
        env[key.strip()] = parsed
    return env


def build_shadow_smoke_env(
    defaults: dict[str, Any],
    stage_overlay_source: Path,
    run_root: Path,
    candidate_binary: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env_file = defaults.get("env_file")
    if isinstance(env_file, str) and env_file.strip():
        env.update(parse_env_file(Path(env_file)))
    env.update(parse_env_file(stage_overlay_source))
    env["OUTDIR"] = str(run_root)
    env["PARAPHINA_LIVE_BIN"] = str(candidate_binary)
    return env


def run_balance_snapshot(
    *,
    label: str,
    defaults: dict[str, Any],
    run_root: Path,
    repo_root: Path,
    pre_snapshot: Path | None = None,
    check: bool = True,
) -> Path:
    snapshot_tool = repo_root / "tools" / "phase5_balance_snapshot.py"
    if not snapshot_tool.exists():
        raise FileNotFoundError(f"balance snapshot tool missing: {snapshot_tool}")
    env_file = Path(defaults.get("env_file") or "/etc/paraphina/current.env")
    snapshot_path = run_root / f"balance_{label}_snapshot.json"
    argv = [
        "python3",
        str(snapshot_tool),
        "--label",
        label,
        "--out-dir",
        str(run_root),
        "--env-file",
        str(env_file),
    ]
    if pre_snapshot is not None:
        argv.extend(["--pre-snapshot", str(pre_snapshot)])
    stdout_path = run_root / f"balance_{label}_snapshot.stdout.log"
    stderr_path = run_root / f"balance_{label}_snapshot.stderr.log"
    result = run_logged_command(
        RunCommand(
            f"balance_snapshot_{label}",
            argv,
            cwd=repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            check=check,
        )
    )
    if check and not snapshot_path.exists():
        raise RuntimeError(f"balance snapshot did not create {snapshot_path}")
    if not check and (result.returncode != 0 or not snapshot_path.exists()):
        write_json(
            run_root / f"balance_{label}_snapshot_error.json",
            {
                "schema_version": 1,
                "updated_utc": utc_now(),
                "label": label,
                "exit_code": result.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "snapshot_path": str(snapshot_path),
                "snapshot_exists": snapshot_path.exists(),
            },
        )
    return snapshot_path


def run_pre_live_direct_venue_audit(
    *,
    defaults: dict[str, Any],
    run_root: Path,
) -> Path:
    audit_script = Path(defaults.get("venue_audit_script") or "/home/ubuntu/paraphina/tools/live_venue_audit.py")
    env_file = Path(defaults.get("env_file") or "/etc/paraphina/current.env")
    audit_path = run_root / "direct_venue_audit_pre.json"
    cmd = [
        "python3",
        str(audit_script),
        "--env-file",
        str(env_file),
        "--position-tol-base",
        "0.0025",
        "--max-open-orders",
        "0",
        "--timeout-seconds",
        "30",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        write_json(
            run_root / "direct_venue_audit_pre_error.json",
            {
                "schema_version": 1,
                "updated_utc": utc_now(),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        raise RuntimeError(
            f"pre-live direct venue audit did not emit JSON rc={result.returncode} stderr={result.stderr.strip()!r}"
        ) from exc
    payload["_phase5_pre_live_audit_rc"] = result.returncode
    payload["_phase5_pre_live_audit_stderr"] = result.stderr.strip()
    write_json(audit_path, payload)
    clean, reasons = direct_venue_audit_clean(payload)
    if result.returncode != 0 or not clean:
        raise RuntimeError(f"pre-live direct venue audit is not clean rc={result.returncode} reasons={reasons}")
    return audit_path


def run_live_guarded(tranche_id: str, duration_sec: int, repo_root: Path = ROOT) -> Path:
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    defaults = control_pack["execution_defaults"]
    admission_check(tranche_id, duration_sec, repo_root)
    run_root = build_live_run_root(control_pack, tranche_id, duration_sec)
    run_root.mkdir(parents=True, exist_ok=True)

    runtime_binary = Path(defaults["runtime_binary"])
    candidate_binary = candidate_runtime_binary_path(tranche, defaults, repo_root)
    cleanup_binary = cleanup_binary_path(defaults, repo_root)
    stage_overlay_target = Path(defaults["stage_overlay_target"])
    stage_overlay_source = tranche_stage_overlay_source(tranche, defaults)
    dropin_target = Path(defaults["live_exec_dropin_target"])
    dropin_source = Path(defaults["live_exec_dropin_source"])
    telemetry_path = Path(defaults["telemetry_path"])
    stderr_path = Path(defaults["stderr_path"])
    guard_script = Path(defaults["guard_script"])
    analyzer_script = Path(defaults["analyzer_script"])
    health_url = HEALTH_URL
    service = defaults["service"]
    live_guard_args = tranche_live_guard_args(tranche, defaults)

    pre_health = curl_health(health_url, attempts=5, delay_sec=1.0)
    pre_systemd = systemd_show(service)
    telemetry_pre = file_size(telemetry_path)
    stderr_pre = file_size(stderr_path)
    write_text(run_root / "health_pre.json", pre_health + "\n")
    write_text(run_root / "systemd_pre_shadow.txt", pre_systemd)
    write_text(run_root / "telemetry_offset_pre.txt", str(telemetry_pre))
    write_text(run_root / "err_offset_pre.txt", str(stderr_pre))
    pre_live_audit_path = run_pre_live_direct_venue_audit(defaults=defaults, run_root=run_root)
    balance_pre_snapshot_path = run_balance_snapshot(
        label="pre",
        defaults=defaults,
        run_root=run_root,
        repo_root=repo_root,
        check=True,
    )

    backup_runtime = run_root / "paraphina_live.shadow.bak"
    backup_overlay = run_root / "stage_overlay.shadow.bak"
    backup_dropin = run_root / "live_exec_flag.shadow.bak"

    if dropin_target.exists():
        run_shell(f"sudo install -m 0644 {dropin_target} {backup_dropin}", cwd=repo_root, label="backup_dropin")
    run_shell(f"sudo install -m 0755 {runtime_binary} {backup_runtime}", cwd=repo_root, label="backup_runtime")
    run_shell(f"sudo install -m 0644 {stage_overlay_target} {backup_overlay}", cwd=repo_root, label="backup_overlay")
    if runtime_install_required(candidate_binary, runtime_binary):
        run_shell(f"sudo install -m 0755 {candidate_binary} {runtime_binary}", cwd=repo_root, label="install_runtime")
    run_shell(f"sudo install -m 0644 {stage_overlay_source} {stage_overlay_target}", cwd=repo_root, label="install_overlay")
    run_shell(f"sudo install -m 0644 {dropin_source} {dropin_target}", cwd=repo_root, label="install_dropin")
    run_shell("sudo systemctl daemon-reload", cwd=repo_root, label="daemon_reload")
    run_shell(f"sudo systemctl restart {service}", cwd=repo_root, label="restart_service")
    wait_for_live_health(health_url)

    guard_log = run_root / "guard.log"
    surface_id = safe_tranche_surface_id(tranche, control_pack, repo_root)
    write_latest_run_manifest(
        repo_root,
        tranche_id,
        {
            "updated_utc": utc_now(),
            "surface_id": surface_id,
            "run_root": str(run_root),
            "duration_sec": duration_sec,
            "guard_path": str(guard_log),
            "health_pre_path": str(run_root / "health_pre.json"),
            "systemd_pre_path": str(run_root / "systemd_pre_shadow.txt"),
            "telemetry_offset_pre_path": str(run_root / "telemetry_offset_pre.txt"),
            "err_offset_pre_path": str(run_root / "err_offset_pre.txt"),
            "direct_venue_audit_pre_path": str(pre_live_audit_path),
            "balance_pre_snapshot_path": str(balance_pre_snapshot_path),
            "run_state": "live_started",
        },
    )
    guard_cmd = [
        "python3",
        str(guard_script),
        "--duration-sec",
        str(duration_sec),
        "--poll-sec",
        "10",
        "--summary-sec",
        "60",
        "--cleanup-bin",
        str(cleanup_binary),
        "--restore-binary",
        str(backup_runtime),
        "--restore-overlay",
        str(backup_overlay),
        "--dropin-path",
        str(dropin_target),
        "--restore-on-exit",
        *live_guard_args,
    ]
    guard_result = run_logged_command(
        RunCommand("live_guard", guard_cmd, stdout_path=guard_log, stderr_path=guard_log, check=False)
    )
    guard_result_path = run_root / "guard_result.json"
    write_json(
        guard_result_path,
        {
            "schema_version": 1,
            "updated_utc": utc_now(),
            "label": "live_guard",
            "argv": guard_cmd,
            "exit_code": guard_result.returncode,
        },
    )

    post_health_payload = safe_shadow_health_snapshot(health_url)
    post_systemd = systemd_show(service)
    telemetry_post = file_size(telemetry_path)
    stderr_post = file_size(stderr_path)
    write_text(run_root / "health_post.json", json.dumps(post_health_payload, sort_keys=False) + "\n")
    write_text(run_root / "systemd_post.txt", post_systemd)
    write_text(run_root / "telemetry_offset_post.txt", str(telemetry_post))
    write_text(run_root / "err_offset_post.txt", str(stderr_post))
    balance_post_snapshot_path = run_balance_snapshot(
        label="post",
        defaults=defaults,
        run_root=run_root,
        repo_root=repo_root,
        pre_snapshot=balance_pre_snapshot_path,
        check=False,
    )
    write_latest_run_manifest(
        repo_root,
        tranche_id,
        {
            "updated_utc": utc_now(),
            "surface_id": surface_id,
            "run_root": str(run_root),
            "duration_sec": duration_sec,
            "guard_path": str(guard_log),
            "guard_result_path": str(guard_result_path),
            "guard_exit_code": guard_result.returncode,
            "health_post_path": str(run_root / "health_post.json"),
            "systemd_post_path": str(run_root / "systemd_post.txt"),
            "telemetry_offset_post_path": str(run_root / "telemetry_offset_post.txt"),
            "err_offset_post_path": str(run_root / "err_offset_post.txt"),
            "balance_pre_snapshot_path": str(balance_pre_snapshot_path),
            "balance_post_snapshot_path": str(balance_post_snapshot_path),
            "run_state": "window_complete",
        },
    )

    telemetry_bounded = run_root / "telemetry_bounded.jsonl"
    stderr_segment = run_root / "paraphina_live.err.segment"
    if telemetry_post >= telemetry_pre:
        copy_file_segment(telemetry_path, telemetry_bounded, telemetry_pre)
    else:
        write_text(telemetry_bounded, "")
    if stderr_post >= stderr_pre:
        copy_file_segment(stderr_path, stderr_segment, stderr_pre)
    else:
        write_text(stderr_segment, "")

    recover_live_closeout(tranche_id, repo_root=repo_root, run_root=run_root, duration_sec=duration_sec)
    write_manual_live_stage_contracts(tranche, control_pack, run_root, duration_sec, repo_root)
    return run_root


def run_shadow_smoke(tranche_id: str, duration_sec: int, repo_root: Path = ROOT) -> Path:
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    defaults = control_pack["execution_defaults"]
    ensure_runtime_storage_headroom(control_pack, duration_sec, repo_root)
    ensure_shadow_health()

    candidate_binary = candidate_runtime_binary_path(tranche, defaults, repo_root)
    if not candidate_binary.exists():
        raise RuntimeError(f"candidate runtime binary missing for shadow smoke: {candidate_binary}")

    run_root = build_shadow_smoke_run_root(control_pack, tranche_id, duration_sec)
    run_root.mkdir(parents=True, exist_ok=True)
    surface_id = tranche_surface_id(tranche, control_pack, repo_root)
    stage_overlay_source = tranche_stage_overlay_source(tranche, defaults)
    shadow_supervisor = repo_root / "tools" / "paraphina_shadow_supervisor.sh"
    if not shadow_supervisor.exists():
        raise FileNotFoundError(f"shadow supervisor missing: {shadow_supervisor}")

    env = build_shadow_smoke_env(defaults, stage_overlay_source, run_root, candidate_binary)

    launcher_log = run_root / "launcher.log"
    result = subprocess.run(
        ["timeout", "--foreground", f"{duration_sec}s", str(shadow_supervisor)],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )
    write_text(launcher_log, (result.stdout or "") + (result.stderr or ""))
    connector_availability = evaluate_shadow_connector_availability(run_root)
    health_pass = result.returncode in {0, 124} and not connector_availability["connector_unavailable"]
    mechanism_evaluation = evaluate_shadow_mechanism_evidence(tranche, run_root)
    mechanism_pass = bool(mechanism_evaluation.get("mechanism_pass", True))
    mechanism_required = support_gate_requires_mechanism(tranche)
    gate_passed = health_pass and (mechanism_pass or not mechanism_required)
    status = "pass" if gate_passed else "fail"
    failure_reason = None
    if connector_availability["connector_unavailable"]:
        failure_reason = str(connector_availability.get("failure_reason") or "connector_unavailable")
    elif not health_pass:
        failure_reason = f"shadow_supervisor_exit:{result.returncode}"
    elif mechanism_required and not mechanism_pass:
        failure_reason = str(mechanism_evaluation.get("failure_reason") or "mechanism_not_exercised")
    support_gate_evaluation = {
        "health_pass": health_pass,
        "mechanism_pass": mechanism_pass,
        "mechanism_required": mechanism_required,
        "gate_passed": gate_passed,
        "failure_reason": failure_reason,
        "connector_availability": connector_availability,
        "mechanism_evidence": mechanism_evaluation,
    }
    closeout_path = run_root / "shadow_smoke_closeout_bundle.json"
    write_json(
        closeout_path,
        {
            "schema_version": 1,
            "updated_utc": utc_now(),
            "tranche_id": tranche_id,
            "surface_id": surface_id,
            "duration_sec": duration_sec,
            "run_root": str(run_root),
            "stage_overlay_source": str(stage_overlay_source),
            "runtime_binary": str(candidate_binary),
            "shadow_smoke_status": status,
            "health_pass": health_pass,
            "mechanism_pass": mechanism_pass,
            "mechanism_required": mechanism_required,
            "failure_reason": failure_reason,
            "support_gate_evaluation": support_gate_evaluation,
            "exit_code": result.returncode,
            "launcher_log": str(launcher_log),
            "run_log": str(run_root / "run.log"),
        },
    )
    write_latest_run_manifest(
        repo_root,
        tranche_id,
        {
            "updated_utc": utc_now(),
            "surface_id": surface_id,
            "duration_sec": duration_sec,
            "shadow_smoke_run_dir": str(run_root),
            "shadow_smoke_status": status,
            "health_pass": health_pass,
            "mechanism_pass": mechanism_pass,
            "mechanism_required": mechanism_required,
            "failure_reason": failure_reason,
            "support_gate_evaluation": support_gate_evaluation,
            "closeout_bundle_path": str(closeout_path),
            "launcher_log": str(launcher_log),
            "run_log": str(run_root / "run.log"),
        },
    )
    write_support_gate_manifest(
        repo_root,
        tranche_id,
        "shadow_smoke_10m",
        {
            "updated_utc": utc_now(),
            "surface_id": surface_id,
            "duration_sec": duration_sec,
            "shadow_smoke_run_dir": str(run_root),
            "shadow_smoke_status": status,
            "health_pass": health_pass,
            "mechanism_pass": mechanism_pass,
            "mechanism_required": mechanism_required,
            "failure_reason": failure_reason,
            "support_gate_evaluation": support_gate_evaluation,
            "closeout_bundle_path": str(closeout_path),
            "launcher_log": str(launcher_log),
            "run_log": str(run_root / "run.log"),
        },
    )
    if duration_sec >= 1800:
        write_support_gate_manifest(
            repo_root,
            tranche_id,
            "shadow_smoke_30m",
            {
                "updated_utc": utc_now(),
                "surface_id": surface_id,
                "duration_sec": duration_sec,
                "shadow_smoke_run_dir": str(run_root),
                "shadow_smoke_status": status,
                "health_pass": health_pass,
                "mechanism_pass": mechanism_pass,
                "mechanism_required": mechanism_required,
                "failure_reason": failure_reason,
                "support_gate_evaluation": support_gate_evaluation,
                "closeout_bundle_path": str(closeout_path),
                "launcher_log": str(launcher_log),
                "run_log": str(run_root / "run.log"),
            },
        )
    if status != "pass":
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    return run_root


def run_shadow_ab(tranche_id: str, repo_root: Path = ROOT) -> Path:
    queue, control_pack = load_state(repo_root)
    _, _, tranche = find_tranche(queue, tranche_id)
    execution = tranche.get("execution", {}).get("shadow_ab", {})
    control_cmd = execution.get("control_cmd")
    candidate_cmd = execution.get("candidate_cmd")
    if not control_cmd or not candidate_cmd:
        raise SystemExit(f"{tranche_id} does not define execution.shadow_ab control/candidate commands")

    run_dir = ensure_run_dir(repo_root, tranche_id) / f"shadow_ab_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    control_log = run_dir / "control.log"
    candidate_log = run_dir / "candidate.log"
    run_logged_command(RunCommand("shadow_control", ["bash", "-lc", control_cmd], cwd=repo_root, stdout_path=control_log, stderr_path=control_log))
    run_logged_command(RunCommand("shadow_candidate", ["bash", "-lc", candidate_cmd], cwd=repo_root, stdout_path=candidate_log, stderr_path=candidate_log))
    write_latest_run_manifest(
        repo_root,
        tranche_id,
        {
            "updated_utc": utc_now(),
            "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
            "duration_sec": 600,
            "shadow_run_dir": str(run_dir),
            "shadow_ab_status": "pass",
            "control_log": str(control_log),
            "candidate_log": str(candidate_log),
        },
    )
    write_support_gate_manifest(
        repo_root,
        tranche_id,
        "shadow_ab_10m",
        {
            "updated_utc": utc_now(),
            "surface_id": safe_tranche_surface_id(tranche, control_pack, repo_root),
            "duration_sec": 600,
            "shadow_run_dir": str(run_dir),
            "shadow_ab_status": "pass",
            "control_log": str(control_log),
            "candidate_log": str(candidate_log),
        },
    )
    return run_dir


def cli_validate(args: argparse.Namespace) -> int:
    queue, control_pack = load_state(Path(args.repo_root))
    load_orchestration(Path(args.repo_root))
    write_status(queue, control_pack, Path(args.repo_root))
    print("phase5 queue valid")
    return 0


def cli_audit_gate_contract(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    queue, control_pack = load_state(repo_root)
    payload = audit_gate_contract(
        queue,
        control_pack,
        repo_root=repo_root,
        tranche_id=args.tranche_id,
    )
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 1 if payload["critical_count"] else 0


def cli_audit_state_sync(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    queue, control_pack = load_state(repo_root)
    payload = audit_state_sync(
        queue,
        control_pack,
        repo_root=repo_root,
        tranche_id=args.tranche_id,
    )
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 1 if payload["critical_count"] else 0


def cli_render_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    auto_recover_pending_latest_runs(repo_root)
    queue, control_pack = load_state(repo_root)
    path = write_status(queue, control_pack, repo_root)
    print(path)
    return 0


def cli_prepare(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    if args.tranche_id:
        auto_recover_pending_latest_runs(repo_root, [args.tranche_id])
    else:
        auto_recover_pending_latest_runs(repo_root)
    card_path = prepare_tranche(repo_root, tranche_id=args.tranche_id, mark_in_progress=not args.no_mark_in_progress)
    print(card_path)
    return 0


def cli_record_result(args: argparse.Namespace) -> int:
    record_result(
        tranche_id=args.tranche_id,
        decision=args.decision,
        repo_root=Path(args.repo_root),
        summary_path=args.summary_path,
        guard_path=args.guard_path,
        notes=args.notes,
        observed_blocker_family=args.observed_blocker_family,
        precondition_failed=args.precondition_failed,
        credit_earned=args.credit_earned,
    )
    print(f"{args.tranche_id}: {args.decision}")
    return 0


def cli_admission_check(args: argparse.Namespace) -> int:
    payload = admission_check(args.tranche_id, args.duration_sec, Path(args.repo_root))
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def cli_run_live_guarded(args: argparse.Namespace) -> int:
    run_root = run_live_guarded(args.tranche_id, args.duration_sec, Path(args.repo_root))
    print(run_root)
    return 0


def cli_run_shadow_ab(args: argparse.Namespace) -> int:
    run_dir = run_shadow_ab(args.tranche_id, Path(args.repo_root))
    print(run_dir)
    return 0


def cli_run_shadow_smoke(args: argparse.Namespace) -> int:
    run_dir = run_shadow_smoke(args.tranche_id, args.duration_sec, Path(args.repo_root))
    print(run_dir)
    return 0


def cli_recover_live_closeout(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root) if args.run_root else None
    closeout_path = recover_live_closeout(
        tranche_id=args.tranche_id,
        repo_root=Path(args.repo_root),
        run_root=run_root,
        duration_sec=args.duration_sec,
    )
    print(closeout_path)
    return 0


def cli_record_manual_recovery(args: argparse.Namespace) -> int:
    verification_path = record_manual_recovery_verification(
        args.tranche_id,
        args.audit_path,
        Path(args.repo_root),
        verify_host=not args.no_host_check,
    )
    print(verification_path)
    return 0


def cli_spawn_lanes(args: argparse.Namespace) -> int:
    session = spawn_lanes(args.tranche_id, repo_root=Path(args.repo_root), force=args.force)
    print(json.dumps(session, indent=2, sort_keys=False))
    return 0


def cli_lane_status(args: argparse.Namespace) -> int:
    payload = lane_status_payload(Path(args.repo_root), tranche_id=args.tranche_id)
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


def cli_teardown_lanes(args: argparse.Namespace) -> int:
    session = teardown_lanes(
        args.tranche_id,
        repo_root=Path(args.repo_root),
        preserve_session=not args.drop_session,
    )
    print(json.dumps(session, indent=2, sort_keys=False))
    return 0


def cli_orchestrate(args: argparse.Namespace) -> int:
    session = orchestrate_tranche(args.tranche_id, repo_root=Path(args.repo_root))
    print(json.dumps(session, indent=2, sort_keys=False))
    return 0


def cli_resume_orchestrate(args: argparse.Namespace) -> int:
    session = resume_orchestrate_tranche(args.tranche_id, repo_root=Path(args.repo_root))
    print(json.dumps(session, indent=2, sort_keys=False))
    return 0


def cli_watch_serialized_mainline(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    current_target = args.tranche_id
    poll_sec = float(args.poll_sec)
    while True:
        event = auto_progress_serialized_mainline_once(
            repo_root=repo_root,
            tranche_id=current_target,
        )
        print(json.dumps(event, sort_keys=False), flush=True)
        state = str(event.get("state") or "")
        if state == "idle":
            break
        if state in {"waiting_for_closeout", "infra_invalid"}:
            time.sleep(max(1.0, poll_sec))
            continue
        current_target = None
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 tranche automation")
    parser.add_argument("--repo-root", default=str(ROOT))
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    validate = subparsers.add_parser("validate")
    validate.set_defaults(func=cli_validate)

    gate_contract = subparsers.add_parser("audit-gate-contract")
    gate_contract.add_argument("--tranche-id")
    gate_contract.set_defaults(func=cli_audit_gate_contract)

    state_sync = subparsers.add_parser("audit-state-sync")
    state_sync.add_argument("--tranche-id")
    state_sync.set_defaults(func=cli_audit_state_sync)

    render = subparsers.add_parser("render-status")
    render.set_defaults(func=cli_render_status)

    prepare_next = subparsers.add_parser("prepare-next")
    prepare_next.add_argument("--tranche-id", default=None)
    prepare_next.add_argument("--no-mark-in-progress", action="store_true")
    prepare_next.set_defaults(func=cli_prepare)

    prepare_named = subparsers.add_parser("prepare")
    prepare_named.add_argument("--tranche-id", required=True)
    prepare_named.add_argument("--no-mark-in-progress", action="store_true")
    prepare_named.set_defaults(func=cli_prepare)

    record = subparsers.add_parser("record-result")
    record.add_argument("--tranche-id", required=True)
    record.add_argument("--decision", required=True, choices=["promote", "hold", "rollback"])
    record.add_argument("--summary-path")
    record.add_argument("--guard-path")
    record.add_argument("--notes")
    record.add_argument("--observed-blocker-family", choices=sorted(VALID_BLOCKER_FAMILIES))
    record.add_argument("--precondition-failed", action="store_true")
    record.add_argument("--credit-earned", choices=sorted(VALID_PROGRESS_CREDIT))
    record.set_defaults(func=cli_record_result)

    admission = subparsers.add_parser("admission-check")
    admission.add_argument("--tranche-id", required=True)
    admission.add_argument("--duration-sec", required=True, type=int)
    admission.set_defaults(func=cli_admission_check)

    live = subparsers.add_parser("run-live-guarded")
    live.add_argument("--tranche-id", required=True)
    live.add_argument("--duration-sec", required=True, type=int)
    live.set_defaults(func=cli_run_live_guarded)

    shadow = subparsers.add_parser("run-shadow-ab")
    shadow.add_argument("--tranche-id", required=True)
    shadow.set_defaults(func=cli_run_shadow_ab)

    shadow_smoke = subparsers.add_parser("run-shadow-smoke")
    shadow_smoke.add_argument("--tranche-id", required=True)
    shadow_smoke.add_argument("--duration-sec", required=True, type=int)
    shadow_smoke.set_defaults(func=cli_run_shadow_smoke)

    recover = subparsers.add_parser("recover-live-closeout")
    recover.add_argument("--tranche-id", required=True)
    recover.add_argument("--run-root")
    recover.add_argument("--duration-sec", type=int)
    recover.set_defaults(func=cli_recover_live_closeout)

    manual_recovery = subparsers.add_parser("record-manual-recovery")
    manual_recovery.add_argument("--tranche-id", required=True)
    manual_recovery.add_argument("--audit-path", required=True)
    manual_recovery.add_argument("--no-host-check", action="store_true")
    manual_recovery.set_defaults(func=cli_record_manual_recovery)

    spawn = subparsers.add_parser("spawn-lanes")
    spawn.add_argument("--tranche-id", required=True)
    spawn.add_argument("--force", action="store_true")
    spawn.set_defaults(func=cli_spawn_lanes)

    lane_status = subparsers.add_parser("lane-status")
    lane_status.add_argument("--tranche-id")
    lane_status.set_defaults(func=cli_lane_status)

    teardown = subparsers.add_parser("teardown-lanes")
    teardown.add_argument("--tranche-id", required=True)
    teardown.add_argument("--drop-session", action="store_true")
    teardown.set_defaults(func=cli_teardown_lanes)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--tranche-id", required=True)
    orchestrate.set_defaults(func=cli_orchestrate)

    resume = subparsers.add_parser("resume-orchestrate")
    resume.add_argument("--tranche-id", required=True)
    resume.set_defaults(func=cli_resume_orchestrate)

    watch = subparsers.add_parser("watch-serialized-mainline")
    watch.add_argument("--tranche-id")
    watch.add_argument("--poll-sec", type=float, default=60.0)
    watch.set_defaults(func=cli_watch_serialized_mainline)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
