#!/usr/bin/env python3
"""Validate V2 paper-only admission decision evidence.

This validator is intentionally separate from the strict shadow validator.
It accepts only the first authority tranche: paper-only filtering of existing
baseline MM intents. It rejects live/capital authority, synthesized intent
authority, false blocker clearance, and raw identifier leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_EVENT_TYPE = "V2_ADMISSION_DECISION"
ALLOWED_STATUS = {"ADMITTED", "HOLD"}
RAW_MARKERS = (
    "client_order_id",
    "venue_order_id",
    "order_id",
    "trade_id",
    "fill_id",
    "private_key",
    "signature",
    "auth_token",
    "secret",
    "raw-",
)


class V2AuthorityValidationError(ValueError):
    pass


@dataclass
class V2AuthoritySummary:
    row_count: int = 0
    admitted_rows: int = 0
    hold_rows: int = 0
    output_intent_count_total: int = 0
    suppressed_mm_intent_count_total: int = 0
    blocker_cleared_any: bool = False
    pressure_complete_claim_any: bool = False
    can_create_new_intents_any: bool = False
    can_mutate_live_orders_any: bool = False

    def to_manifest_validation(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "admitted_rows": self.admitted_rows,
            "hold_rows": self.hold_rows,
            "output_intent_count_total": self.output_intent_count_total,
            "suppressed_mm_intent_count_total": self.suppressed_mm_intent_count_total,
            "blocker_cleared_any": self.blocker_cleared_any,
            "pressure_complete_claim_any": self.pressure_complete_claim_any,
            "can_create_new_intents_any": self.can_create_new_intents_any,
            "can_mutate_live_orders_any": self.can_mutate_live_orders_any,
        }


def _require(condition: bool, line: int, message: str) -> None:
    if not condition:
        raise V2AuthorityValidationError(f"line {line}: {message}")


def _file_info(path: Path, artifact_root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        rel = path.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError:
        rel = path.name
    return {
        "path": rel,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _contains_raw_marker(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True).lower()
    return any(marker in text for marker in RAW_MARKERS)


def _validate_gate_state(row: dict[str, Any], line: int) -> bool:
    gate_state = row.get("gate_state")
    _require(isinstance(gate_state, dict), line, "gate_state must be object")
    expected_bool_fields = [
        "enabled",
        "decision_mode_is_paper_admission",
        "execution_mode_is_paper",
        "pair_edge_enabled",
        "pair_conditioned_admission_enabled",
        "order_intent_enabled",
        "fast_hedge_disabled",
        "require_phase51_gate",
    ]
    for field in expected_bool_fields:
        _require(isinstance(gate_state.get(field), bool), line, f"gate_state.{field} must be bool")
    return all(gate_state[field] for field in expected_bool_fields)


def _validate_pair_edges(row: dict[str, Any], line: int) -> None:
    pair_edges = row.get("pair_edges")
    _require(isinstance(pair_edges, list), line, "pair_edges must be list")
    for idx, pair_edge in enumerate(pair_edges):
        _require(isinstance(pair_edge, dict), line, f"pair_edges[{idx}] must be object")
        _require(pair_edge.get("feature_only") is False, line, f"pair_edges[{idx}] must be admission-scoped")
        edge_usd = pair_edge.get("edge_usd")
        _require(edge_usd is None or isinstance(edge_usd, (int, float)), line, f"pair_edges[{idx}] edge_usd invalid")
        for ref_field in ("bid_candidate_id", "ask_candidate_id"):
            value = pair_edge.get(ref_field)
            _require(value is None or isinstance(value, str), line, f"pair_edges[{idx}] {ref_field} invalid")


def _validate_admitted_candidates(row: dict[str, Any], line: int, gate_satisfied: bool) -> None:
    candidates = row.get("admitted_candidates")
    _require(isinstance(candidates, list), line, "admitted_candidates must be list")
    if row.get("admission_status") == "ADMITTED":
        _require(gate_satisfied, line, "ADMITTED row without satisfied gate")
        _require(candidates, line, "ADMITTED row without candidates")
    else:
        _require(not candidates, line, "HOLD row must not admit candidates")
    for idx, candidate in enumerate(candidates):
        _require(isinstance(candidate, dict), line, f"admitted_candidates[{idx}] must be object")
        candidate_id = candidate.get("candidate_id")
        _require(
            isinstance(candidate_id, str) and candidate_id.startswith("v2_shadow_intent_v1:"),
            line,
            f"admitted_candidates[{idx}] candidate_id invalid",
        )
        _require(isinstance(candidate.get("venue_index"), int), line, f"admitted_candidates[{idx}] venue_index invalid")
        _require(isinstance(candidate.get("venue_id"), str), line, f"admitted_candidates[{idx}] venue_id invalid")
        _require(candidate.get("side") in {"Buy", "Sell"}, line, f"admitted_candidates[{idx}] side invalid")
        _require(isinstance(candidate.get("rank_index"), int), line, f"admitted_candidates[{idx}] rank_index invalid")
        _require(
            isinstance(candidate.get("rank_score_microusd"), int)
            and candidate["rank_score_microusd"] > 0,
            line,
            f"admitted_candidates[{idx}] rank_score_microusd must be positive",
        )


def _validate_row(row: dict[str, Any], line: int, summary: V2AuthoritySummary) -> None:
    _require(row.get("event_type") == EXPECTED_EVENT_TYPE, line, "event_type invalid")
    _require(row.get("schema_version") == 1, line, "schema_version must be 1")
    _require(row.get("decision_mode") == "paper_admission", line, "decision_mode must be paper_admission")
    _require(row.get("execution_mode") == "paper", line, "execution_mode must be paper")
    _require(row.get("authority_scope") == "paper_only", line, "authority_scope must be paper_only")
    _require(row.get("admission_status") in ALLOWED_STATUS, line, "admission_status invalid")
    _require(row.get("can_create_new_intents") is False, line, "can_create_new_intents must be false")
    _require(row.get("can_mutate_live_orders") is False, line, "can_mutate_live_orders must be false")
    _require(row.get("ranking_feature_only") is False, line, "ranking_feature_only must be false")
    _require(row.get("pressure_complete_claim") is False, line, "pressure_complete_claim must be false")
    _require(row.get("blocker_cleared") is False, line, "blocker_cleared must be false")
    _require(isinstance(row.get("order_intent_output_count"), int), line, "order_intent_output_count invalid")
    _require(
        row["order_intent_output_count"] <= row.get("baseline_mm_order_creating_intent_count", -1),
        line,
        "V2 authority cannot increase baseline MM order-creating intent count",
    )
    _require(not _contains_raw_marker(row), line, "raw identifier or secret marker present")

    gate_satisfied = _validate_gate_state(row, line)
    _require(
        row.get("can_filter_existing_intents") is gate_satisfied,
        line,
        "can_filter_existing_intents must match satisfied gate state",
    )
    if row.get("admission_status") == "ADMITTED":
        _require(row.get("pair_edge_is_admission") is True, line, "ADMITTED row must use pair edge as admission")
        _require(row.get("ranking_is_admission") is True, line, "ADMITTED row must use ranking as admission")
    else:
        _require(row.get("order_intent_output_count") == 0, line, "HOLD row must output zero intents")
    _validate_pair_edges(row, line)
    _validate_admitted_candidates(row, line, gate_satisfied)

    summary.row_count += 1
    if row["admission_status"] == "ADMITTED":
        summary.admitted_rows += 1
    else:
        summary.hold_rows += 1
    summary.output_intent_count_total += row["order_intent_output_count"]
    summary.suppressed_mm_intent_count_total += row.get("suppressed_mm_order_creating_intent_count", 0)
    summary.blocker_cleared_any = summary.blocker_cleared_any or row["blocker_cleared"]
    summary.pressure_complete_claim_any = summary.pressure_complete_claim_any or row["pressure_complete_claim"]
    summary.can_create_new_intents_any = summary.can_create_new_intents_any or row["can_create_new_intents"]
    summary.can_mutate_live_orders_any = summary.can_mutate_live_orders_any or row["can_mutate_live_orders"]


def validate_v2_authority_decisions(path: Path) -> V2AuthoritySummary:
    summary = V2AuthoritySummary()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise V2AuthorityValidationError(f"line {line_no}: invalid JSON: {exc}") from exc
            _require(isinstance(row, dict), line_no, "row must be object")
            _validate_row(row, line_no, summary)
    if summary.row_count == 0:
        raise V2AuthorityValidationError("no V2 authority decision rows found")
    return summary


def write_manifest(decision_path: Path, manifest_path: Path, summary: V2AuthoritySummary) -> None:
    artifact_root = manifest_path.parent
    manifest = {
        "artifact_type": "v2_authority_decision_evidence_manifest",
        "schema_version": 1,
        "decision_validation_status": "pass",
        "files": [_file_info(decision_path, artifact_root)],
        "governance": {
            "gate_status": "PAPER_ONLY",
            "shadow_only": False,
            "paper_only": True,
            "approved_for_live": False,
            "approved_for_canary": False,
            "approved_for_capital_escalation": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "blocker_cleared": False,
            "pressure_complete_claim": False,
        },
        "v2_authority_contract": {
            "authority_scope": "paper_only",
            "can_filter_existing_intents": True,
            "can_create_new_intents": False,
            "can_mutate_live_orders": False,
            "baseline_intent_filter_only": True,
        },
        "validation": summary.to_manifest_validation(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-authority-decisions", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_v2_authority_decisions(args.v2_authority_decisions)
        if args.manifest_output:
            write_manifest(args.v2_authority_decisions, args.manifest_output, summary)
    except V2AuthorityValidationError as exc:
        print(f"V2_AUTHORITY_DECISION_VALIDATION_FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"V2_AUTHORITY_DECISION_VALIDATOR_ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "V2_AUTHORITY_DECISION_VALIDATION_PASS "
        f"rows={summary.row_count} admitted_rows={summary.admitted_rows} "
        f"output_intent_count={summary.output_intent_count_total}"
    )
    if args.manifest_output:
        print(f"V2_AUTHORITY_DECISION_MANIFEST_WRITTEN path={args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
