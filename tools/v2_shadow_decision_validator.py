#!/usr/bin/env python3
"""Validate V2 shadow-only decision evidence.

The V2 shadow decision sidecar is admissible only as observation evidence. It
must never become order authority, pressure-complete evidence, or blocker
clearance. This validator enforces that contract and can write a small evidence
manifest for reviewed shadow runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_EVENT_TYPE = "V2_SHADOW_DECISION"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_TELEMETRY_SCHEMA_VERSION = 2
EXPECTED_DECISION_MODE = "shadow"
EXPECTED_ADMISSION_STATUS = "HOLD"
EXPECTED_ADMISSION_REASON = "shadow_only_no_order_authority"
ALLOWED_TARGET_LINKAGE_STATES = {"missing", "present_redacted"}
ALLOWED_SIDES = {"Buy", "Sell"}
ALLOWED_PAIR_EDGE_INVALID_REASONS = {None, "missing_bid", "missing_ask"}
ALLOWED_RANKING_STATUSES = {"scored", "missing_cross_venue_reference"}
CANDIDATE_ID_PREFIXES = ("v2_shadow_v1:", "v2_shadow_intent_v1:")
PAIR_EDGE_ID_PREFIX = "v2_pair_edge_v1:"

FORBIDDEN_KEY_NAMES = {
    "account_id",
    "account_l1_address",
    "approved",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_live",
    "api_key",
    "auth_token",
    "capital_change_allowed",
    "canonical_group_id",
    "cloid",
    "client_order_id",
    "execution_priority",
    "fill_id",
    "headers",
    "live_orders_allowed",
    "oid",
    "order_intent",
    "order_id",
    "order_key",
    "private_key",
    "raw_payload",
    "raw_request",
    "raw_response",
    "ranked_order_intents",
    "secret",
    "selected",
    "selected_candidate_id",
    "signature",
    "tid",
    "token",
    "trade_id",
    "venue_order_id",
    "wallet_address",
    "winner",
    "volume_quota_remaining",
}
FORBIDDEN_KEY_PREFIXES = ("raw_",)
FORBIDDEN_STRING_FRAGMENTS = (
    "api_key",
    "auth_token",
    "canonical_group_id",
    "client_order_id",
    "execution_priority",
    "order_id",
    "order_key",
    "ranked_order_intents",
    "private_key",
    "raw-client",
    "raw-group",
    "raw-order",
    "secret",
    "selected_candidate",
    "signature",
    "token",
    "venue_order_id",
    "volume_quota_remaining",
)


class ContractViolation(Exception):
    """Raised when evidence violates the V2 shadow decision contract."""


@dataclass
class ValidationSummary:
    row_count: int = 0
    baseline_plan_intent_count_total: int = 0
    baseline_mm_order_creating_intent_count_total: int = 0
    rows_with_baseline_mm_order_creating_intents: int = 0
    candidate_count_total: int = 0
    rows_with_candidates: int = 0
    candidate_ranking_count_total: int = 0
    rows_with_candidate_rankings: int = 0
    pair_edge_count_total: int = 0
    rows_with_pair_edges: int = 0
    can_mutate_orders_any: bool = False
    blocker_cleared_any: bool = False
    pressure_complete_claim_any: bool = False
    order_intent_output_count_total: int = 0
    candidate_target_linkage_states: dict[str, int] = field(default_factory=dict)

    def as_manifest_payload(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "baseline_plan_intent_count_total": self.baseline_plan_intent_count_total,
            "baseline_mm_order_creating_intent_count_total": (
                self.baseline_mm_order_creating_intent_count_total
            ),
            "rows_with_baseline_mm_order_creating_intents": (
                self.rows_with_baseline_mm_order_creating_intents
            ),
            "candidate_count_total": self.candidate_count_total,
            "rows_with_candidates": self.rows_with_candidates,
            "candidate_ranking_count_total": self.candidate_ranking_count_total,
            "rows_with_candidate_rankings": self.rows_with_candidate_rankings,
            "pair_edge_count_total": self.pair_edge_count_total,
            "rows_with_pair_edges": self.rows_with_pair_edges,
            "can_mutate_orders_any": self.can_mutate_orders_any,
            "blocker_cleared_any": self.blocker_cleared_any,
            "pressure_complete_claim_any": self.pressure_complete_claim_any,
            "order_intent_output_count_total": self.order_intent_output_count_total,
            "candidate_target_linkage_states": dict(
                sorted(self.candidate_target_linkage_states.items())
            ),
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative_artifact_path(path: Path, artifact_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = artifact_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as err:
        raise ValueError(f"artifact must be under manifest root: {path}") from err
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe artifact path: {path}")
    return relative.as_posix()


def _file_info(path: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "path": _relative_artifact_path(path, artifact_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(condition: bool, line: int, message: str) -> None:
    if not condition:
        raise ContractViolation(f"line {line}: {message}")


def _require_exact(row: dict[str, Any], field_name: str, expected: Any, line: int) -> None:
    actual = row.get(field_name)
    _require(
        actual == expected,
        line,
        f"{field_name} expected {expected!r}, got {actual!r}",
    )


def _scan_for_forbidden_material(value: Any, line: int, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            _require(
                lowered not in FORBIDDEN_KEY_NAMES,
                line,
                f"forbidden field name at {path}.{key}: {key}",
            )
            _require(
                not any(lowered.startswith(prefix) for prefix in FORBIDDEN_KEY_PREFIXES),
                line,
                f"forbidden raw field prefix at {path}.{key}: {key}",
            )
            _scan_for_forbidden_material(child, line, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _scan_for_forbidden_material(child, line, f"{path}[{idx}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_STRING_FRAGMENTS:
            _require(
                fragment not in lowered,
                line,
                f"forbidden string material at {path}: contains {fragment}",
            )


def _validate_candidate(candidate: Any, line: int, index: int) -> str:
    _require(isinstance(candidate, dict), line, f"candidate[{index}] must be object")
    candidate_id = candidate.get("candidate_id")
    _require(isinstance(candidate_id, str) and candidate_id, line, f"candidate[{index}] missing id")
    _require(
        candidate_id.startswith(CANDIDATE_ID_PREFIXES),
        line,
        f"candidate[{index}] id has unsupported prefix",
    )
    _require(_is_int(candidate.get("venue_index")), line, f"candidate[{index}] venue_index invalid")
    _require(candidate["venue_index"] >= 0, line, f"candidate[{index}] venue_index negative")
    _require(
        isinstance(candidate.get("venue_id"), str) and candidate["venue_id"],
        line,
        f"candidate[{index}] venue_id invalid",
    )
    _require(candidate.get("side") in ALLOWED_SIDES, line, f"candidate[{index}] side invalid")
    _require(_is_finite_number(candidate.get("price")), line, f"candidate[{index}] price invalid")
    _require(_is_finite_number(candidate.get("size")), line, f"candidate[{index}] size invalid")
    _require(candidate["size"] > 0, line, f"candidate[{index}] size must be positive")
    linkage_state = candidate.get("target_linkage_state")
    _require(
        linkage_state in ALLOWED_TARGET_LINKAGE_STATES,
        line,
        f"candidate[{index}] target_linkage_state invalid",
    )
    _require_exact(candidate, "admission_status", EXPECTED_ADMISSION_STATUS, line)
    _require_exact(candidate, "admission_reason", EXPECTED_ADMISSION_REASON, line)
    return str(linkage_state)


def _validate_pair_edge(pair_edge: Any, line: int, index: int, candidate_ids: set[str]) -> None:
    _require(isinstance(pair_edge, dict), line, f"pair_edges[{index}] must be object")
    snapshot_id = pair_edge.get("snapshot_id")
    _require(
        isinstance(snapshot_id, str) and snapshot_id.startswith(PAIR_EDGE_ID_PREFIX),
        line,
        f"pair_edges[{index}] snapshot_id invalid",
    )
    for field_name in ("bid_candidate_id", "ask_candidate_id"):
        value = pair_edge.get(field_name)
        _require(
            value is None
            or (isinstance(value, str) and value.startswith(CANDIDATE_ID_PREFIXES)),
            line,
            f"pair_edges[{index}] {field_name} invalid",
        )
        _require(
            value is None or value in candidate_ids,
            line,
            f"pair_edges[{index}] {field_name} does not reference emitted candidate",
        )
    for field_name in ("edge_usd", "edge_bps"):
        value = pair_edge.get(field_name)
        _require(
            value is None or _is_finite_number(value),
            line,
            f"pair_edges[{index}] {field_name} invalid",
        )
    _require(pair_edge.get("feature_only") is True, line, f"pair_edges[{index}] not feature_only")
    _require(
        pair_edge.get("invalid_reason") in ALLOWED_PAIR_EDGE_INVALID_REASONS,
        line,
        f"pair_edges[{index}] invalid_reason unsupported",
    )


def _validate_candidate_ranking(
    ranking: Any,
    line: int,
    index: int,
    candidate_ids: set[str],
    seen_rank_indexes: set[int],
    seen_ranked_candidate_ids: set[str],
) -> None:
    _require(isinstance(ranking, dict), line, f"candidate_rankings[{index}] must be object")
    _require(_is_int(ranking.get("rank_index")), line, f"candidate_rankings[{index}] rank_index invalid")
    _require(ranking["rank_index"] > 0, line, f"candidate_rankings[{index}] rank_index must be positive")
    _require(
        ranking["rank_index"] not in seen_rank_indexes,
        line,
        f"candidate_rankings[{index}] duplicate rank_index: {ranking['rank_index']}",
    )
    seen_rank_indexes.add(ranking["rank_index"])
    candidate_id = ranking.get("candidate_id")
    _require(
        isinstance(candidate_id, str) and candidate_id in candidate_ids,
        line,
        f"candidate_rankings[{index}] candidate_id does not reference emitted candidate",
    )
    _require(
        candidate_id not in seen_ranked_candidate_ids,
        line,
        f"candidate_rankings[{index}] duplicate candidate_id: {candidate_id}",
    )
    seen_ranked_candidate_ids.add(candidate_id)
    _require(
        ranking.get("rank_status") in ALLOWED_RANKING_STATUSES,
        line,
        f"candidate_rankings[{index}] rank_status invalid",
    )
    _require(
        _is_int(ranking.get("rank_score_microusd")),
        line,
        f"candidate_rankings[{index}] rank_score_microusd invalid",
    )
    feature_usd = ranking.get("pair_edge_feature_usd")
    feature_bps = ranking.get("pair_edge_feature_bps")
    _require(
        feature_usd is None or _is_finite_number(feature_usd),
        line,
        f"candidate_rankings[{index}] pair_edge_feature_usd invalid",
    )
    _require(
        feature_bps is None or _is_finite_number(feature_bps),
        line,
        f"candidate_rankings[{index}] pair_edge_feature_bps invalid",
    )
    for field_name in ("reference_candidate_id", "reference_venue_id"):
        value = ranking.get(field_name)
        _require(value is None or isinstance(value, str), line, f"candidate_rankings[{index}] {field_name} invalid")
    reference_candidate_id = ranking.get("reference_candidate_id")
    _require(
        reference_candidate_id is None or reference_candidate_id in candidate_ids,
        line,
        f"candidate_rankings[{index}] reference_candidate_id does not reference emitted candidate",
    )
    reference_venue_index = ranking.get("reference_venue_index")
    _require(
        reference_venue_index is None or (_is_int(reference_venue_index) and reference_venue_index >= 0),
        line,
        f"candidate_rankings[{index}] reference_venue_index invalid",
    )
    _require(
        isinstance(ranking.get("rank_tiebreak_key"), str) and ranking["rank_tiebreak_key"],
        line,
        f"candidate_rankings[{index}] rank_tiebreak_key invalid",
    )
    if ranking["rank_status"] == "scored":
        _require(reference_candidate_id is not None, line, f"candidate_rankings[{index}] scored rank missing reference")
        _require(feature_usd is not None, line, f"candidate_rankings[{index}] scored rank missing feature")
    else:
        _require(reference_candidate_id is None, line, f"candidate_rankings[{index}] missing-reference rank has reference")
        _require(feature_usd is None, line, f"candidate_rankings[{index}] missing-reference rank has feature")
    _require(ranking.get("feature_only") is True, line, f"candidate_rankings[{index}] not feature_only")
    _require_exact(ranking, "admission_status", EXPECTED_ADMISSION_STATUS, line)
    _require_exact(ranking, "admission_reason", EXPECTED_ADMISSION_REASON, line)


def _validate_row(row: Any, line: int, summary: ValidationSummary) -> None:
    _require(isinstance(row, dict), line, "row must be object")
    _scan_for_forbidden_material(row, line)

    _require_exact(row, "event_type", EXPECTED_EVENT_TYPE, line)
    _require_exact(row, "schema_version", EXPECTED_SCHEMA_VERSION, line)
    _require_exact(row, "telemetry_schema_version", EXPECTED_TELEMETRY_SCHEMA_VERSION, line)
    _require(_is_int(row.get("now_ms")), line, "now_ms must be integer")
    _require(row["now_ms"] >= 0, line, "now_ms must be nonnegative")
    _require_exact(row, "decision_mode", EXPECTED_DECISION_MODE, line)
    _require_exact(row, "admission_status", EXPECTED_ADMISSION_STATUS, line)
    _require_exact(row, "admission_reason", EXPECTED_ADMISSION_REASON, line)
    _require(row.get("can_mutate_orders") is False, line, "can_mutate_orders must be false")
    _require(
        row.get("order_intent_output_count") == 0,
        line,
        "order_intent_output_count must remain zero",
    )
    _require(_is_int(row.get("baseline_plan_intent_count")), line, "baseline_plan_intent_count invalid")
    _require(
        _is_int(row.get("baseline_mm_order_creating_intent_count")),
        line,
        "baseline_mm_order_creating_intent_count invalid",
    )
    _require(row["baseline_plan_intent_count"] >= 0, line, "baseline_plan_intent_count negative")
    _require(
        row["baseline_mm_order_creating_intent_count"] >= 0,
        line,
        "baseline_mm_order_creating_intent_count negative",
    )
    _require(
        row["baseline_mm_order_creating_intent_count"] <= row["baseline_plan_intent_count"],
        line,
        "baseline_mm_order_creating_intent_count exceeds baseline_plan_intent_count",
    )
    _require(row.get("pair_edge_is_admission") is False, line, "pair_edge_is_admission must be false")
    _require(row.get("pressure_complete_claim") is False, line, "pressure_complete_claim must be false")
    _require(row.get("blocker_cleared") is False, line, "blocker_cleared must be false")
    _require(row.get("require_phase51_gate") is True, line, "require_phase51_gate must be true")
    _require(
        row.get("pair_conditioned_admission_enabled") is False,
        line,
        "pair_conditioned_admission_enabled must be false",
    )
    _require(row.get("fast_hedge_enabled") is False, line, "fast_hedge_enabled must be false")
    _require(row.get("order_intent_enabled") is False, line, "order_intent_enabled must be false")
    if (
        "ranking_schema_version" in row
        or "ranking_feature_only" in row
        or "candidate_rankings" in row
        or "ranking_is_admission" in row
    ):
        _require_exact(row, "ranking_schema_version", 1, line)
        _require(row.get("ranking_feature_only") is True, line, "ranking_feature_only must be true")
        _require(row.get("ranking_is_admission") is False, line, "ranking_is_admission must be false")

    candidates = row.get("candidates")
    candidate_rankings = row.get("candidate_rankings", [])
    pair_edges = row.get("pair_edges")
    _require(isinstance(candidates, list), line, "candidates must be list")
    _require(isinstance(candidate_rankings, list), line, "candidate_rankings must be list")
    _require(isinstance(pair_edges, list), line, "pair_edges must be list")

    summary.row_count += 1
    summary.baseline_plan_intent_count_total += row["baseline_plan_intent_count"]
    mm_creating_count = row["baseline_mm_order_creating_intent_count"]
    summary.baseline_mm_order_creating_intent_count_total += mm_creating_count
    if mm_creating_count > 0:
        summary.rows_with_baseline_mm_order_creating_intents += 1
    summary.order_intent_output_count_total += row["order_intent_output_count"]
    summary.can_mutate_orders_any = summary.can_mutate_orders_any or row["can_mutate_orders"]
    summary.blocker_cleared_any = summary.blocker_cleared_any or row["blocker_cleared"]
    summary.pressure_complete_claim_any = (
        summary.pressure_complete_claim_any or row["pressure_complete_claim"]
    )

    summary.candidate_count_total += len(candidates)
    if candidates:
        summary.rows_with_candidates += 1
    candidate_ids: set[str] = set()
    for idx, candidate in enumerate(candidates):
        linkage_state = _validate_candidate(candidate, line, idx)
        candidate_id = candidate["candidate_id"]
        _require(candidate_id not in candidate_ids, line, f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        summary.candidate_target_linkage_states[linkage_state] = (
            summary.candidate_target_linkage_states.get(linkage_state, 0) + 1
        )

    summary.candidate_ranking_count_total += len(candidate_rankings)
    if candidate_rankings:
        summary.rows_with_candidate_rankings += 1
        _require(
            len(candidate_rankings) == len(candidates),
            line,
            "candidate_rankings count must equal candidates count",
        )
    seen_rank_indexes: set[int] = set()
    seen_ranked_candidate_ids: set[str] = set()
    for idx, ranking in enumerate(candidate_rankings):
        _validate_candidate_ranking(
            ranking, line, idx, candidate_ids, seen_rank_indexes, seen_ranked_candidate_ids
        )
    if candidate_rankings:
        expected_rank_indexes = set(range(1, len(candidate_rankings) + 1))
        _require(
            seen_rank_indexes == expected_rank_indexes,
            line,
            "candidate_rankings rank_index values must be dense from 1",
        )

    summary.pair_edge_count_total += len(pair_edges)
    if pair_edges:
        summary.rows_with_pair_edges += 1
    for idx, pair_edge in enumerate(pair_edges):
        _validate_pair_edge(pair_edge, line, idx, candidate_ids)


def validate_v2_shadow_decisions(
    path: Path,
    *,
    require_candidate: bool = True,
    require_mm_creating_intent: bool = True,
) -> ValidationSummary:
    if not path.exists():
        raise FileNotFoundError(path)
    summary = ValidationSummary()
    with path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                raise ContractViolation(f"line {line_num}: blank line is not valid JSONL evidence")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ContractViolation(f"line {line_num}: invalid JSON: {err}") from err
            _validate_row(row, line_num, summary)

    if summary.row_count == 0:
        raise ContractViolation("no V2 shadow decision rows found")
    if require_candidate and summary.rows_with_candidates == 0:
        raise ContractViolation("no V2 shadow candidate rows found")
    if require_mm_creating_intent and summary.rows_with_baseline_mm_order_creating_intents == 0:
        raise ContractViolation("no baseline MM order-creating intent observation found")
    return summary


def build_manifest(
    *,
    v2_shadow_decisions: Path,
    telemetry: Path | None,
    summary_path: Path | None,
    validation: ValidationSummary,
    artifact_root: Path,
) -> dict[str, Any]:
    files = [_file_info(v2_shadow_decisions, artifact_root)]
    for optional_path in (telemetry, summary_path):
        if optional_path is not None:
            if not optional_path.exists():
                raise FileNotFoundError(optional_path)
            files.append(_file_info(optional_path, artifact_root))

    return {
        "schema_version": 1,
        "artifact_type": "v2_shadow_decision_evidence_manifest",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "decision_validation_status": "pass",
        "governance": {
            "gate_status": "HOLD",
            "no_live_flag": True,
            "approved_for_live": False,
            "approved_for_canary": False,
            "approved_for_capital_escalation": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
            "blocker_cleared": False,
            "pressure_complete_claim": False,
            "shadow_only": True,
        },
        "v2_shadow_contract": {
            "decision_mode": EXPECTED_DECISION_MODE,
            "admission_status": EXPECTED_ADMISSION_STATUS,
            "admission_reason": EXPECTED_ADMISSION_REASON,
            "can_mutate_orders": False,
            "order_intent_output_count": 0,
            "ranking_schema_version": 1,
            "ranking_feature_only": True,
            "ranking_is_admission": False,
            "pair_edge_is_admission": False,
            "pressure_complete_claim": False,
            "blocker_cleared": False,
            "require_phase51_gate": True,
        },
        "validation": validation.as_manifest_payload(),
        "files": files,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-shadow-decisions", required=True, type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--summary", dest="summary_path", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument(
        "--allow-no-candidates",
        action="store_true",
        help="Permit a valid but candidate-empty shadow run.",
    )
    parser.add_argument(
        "--allow-no-mm-creating-intents",
        action="store_true",
        help="Permit a valid but MM-intent-empty shadow run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        validation = validate_v2_shadow_decisions(
            args.v2_shadow_decisions,
            require_candidate=not args.allow_no_candidates,
            require_mm_creating_intent=not args.allow_no_mm_creating_intents,
        )
        if args.manifest_output is not None:
            manifest = build_manifest(
                v2_shadow_decisions=args.v2_shadow_decisions,
                telemetry=args.telemetry,
                summary_path=args.summary_path,
                validation=validation,
                artifact_root=args.manifest_output.parent,
            )
            write_manifest(args.manifest_output, manifest)
    except ContractViolation as err:
        print(f"V2_SHADOW_DECISION_VALIDATION_FAIL: {err}", file=sys.stderr)
        return 1
    except (OSError, FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        print(f"V2_SHADOW_DECISION_VALIDATOR_ERROR: {err}", file=sys.stderr)
        return 2

    print(
        "V2_SHADOW_DECISION_VALIDATION_PASS "
        f"rows={validation.row_count} "
        f"rows_with_candidates={validation.rows_with_candidates} "
        "rows_with_mm_order_creating_intents="
        f"{validation.rows_with_baseline_mm_order_creating_intents} "
        f"candidate_count={validation.candidate_count_total} "
        f"pair_edge_count={validation.pair_edge_count_total}"
    )
    if args.manifest_output is not None:
        print(f"V2_SHADOW_DECISION_MANIFEST_WRITTEN path={args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
