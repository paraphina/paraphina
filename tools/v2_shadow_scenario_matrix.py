#!/usr/bin/env python3
"""Generate a deterministic V2 shadow-only scenario matrix.

The matrix is an offline evidence pack for exercising V2 shadow candidates and
pair-edge feature snapshots across all configured venue families. It produces
only HOLD/no-order-authority `V2_SHADOW_DECISION` rows and validates the output
with `tools.v2_shadow_decision_validator`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import v2_shadow_decision_validator as validator


VENUES = ("extended", "hyperliquid", "aster", "lighter", "paradex")
EXPECTED_ADMISSION_REASON = "shadow_only_no_order_authority"


def _candidate(
    *,
    venue_index: int,
    venue_id: str,
    side: str,
    price: float,
    size: float,
    target_linkage_state: str = "present_redacted",
) -> dict[str, Any]:
    return {
        "candidate_id": f"v2_shadow_v1:{venue_index}:{venue_id}:{side.lower()}",
        "venue_index": venue_index,
        "venue_id": venue_id,
        "side": side,
        "price": price,
        "size": size,
        "target_linkage_state": target_linkage_state,
        "admission_status": "HOLD",
        "admission_reason": EXPECTED_ADMISSION_REASON,
    }


def _intent_candidate(
    *,
    venue_index: int,
    venue_id: str,
    side: str,
    intent_index: int,
    price: float,
    size: float,
    target_linkage_state: str,
) -> dict[str, Any]:
    return {
        "candidate_id": f"v2_shadow_intent_v1:{venue_index}:{venue_id}:{side}:{intent_index}",
        "venue_index": venue_index,
        "venue_id": venue_id,
        "side": side,
        "price": price,
        "size": size,
        "target_linkage_state": target_linkage_state,
        "admission_status": "HOLD",
        "admission_reason": EXPECTED_ADMISSION_REASON,
    }


def _pair_edge(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    bids = [candidate for candidate in candidates if candidate["side"] == "Buy"]
    asks = [candidate for candidate in candidates if candidate["side"] == "Sell"]
    if not bids:
        return {
            "snapshot_id": "v2_pair_edge_v1:missing_bid",
            "bid_candidate_id": None,
            "ask_candidate_id": min(asks, key=lambda candidate: candidate["price"])["candidate_id"]
            if asks
            else None,
            "edge_usd": None,
            "edge_bps": None,
            "feature_only": True,
            "invalid_reason": "missing_bid",
        }
    if not asks:
        return {
            "snapshot_id": "v2_pair_edge_v1:missing_ask",
            "bid_candidate_id": max(bids, key=lambda candidate: candidate["price"])["candidate_id"],
            "ask_candidate_id": None,
            "edge_usd": None,
            "edge_bps": None,
            "feature_only": True,
            "invalid_reason": "missing_ask",
        }

    best_bid = max(bids, key=lambda candidate: candidate["price"])
    best_ask = min(asks, key=lambda candidate: candidate["price"])
    edge_usd = best_bid["price"] - best_ask["price"]
    midpoint = (best_bid["price"] + best_ask["price"]) / 2.0
    return {
        "snapshot_id": f"v2_pair_edge_v1:{best_bid['candidate_id']}:{best_ask['candidate_id']}",
        "bid_candidate_id": best_bid["candidate_id"],
        "ask_candidate_id": best_ask["candidate_id"],
        "edge_usd": edge_usd,
        "edge_bps": edge_usd / midpoint * 10_000.0,
        "feature_only": True,
        "invalid_reason": None,
    }


def _candidate_rankings(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for candidate in candidates:
        reference = _best_same_side_reference(candidate, candidates)
        if reference is None:
            rank_status = "missing_cross_venue_reference"
            rank_score_microusd = 0
            feature_usd = None
            feature_bps = None
            reference_candidate_id = None
            reference_venue_index = None
            reference_venue_id = None
        else:
            rank_status = "scored"
            feature_usd = (
                reference["price"] - candidate["price"]
                if candidate["side"] == "Buy"
                else candidate["price"] - reference["price"]
            )
            midpoint = (reference["price"] + candidate["price"]) / 2.0
            feature_bps = feature_usd / midpoint * 10_000.0 if midpoint > 0 else None
            rank_score_microusd = round(feature_usd * 1_000_000)
            reference_candidate_id = reference["candidate_id"]
            reference_venue_index = reference["venue_index"]
            reference_venue_id = reference["venue_id"]
        linkage_tiebreak = 0 if candidate["target_linkage_state"] == "present_redacted" else 1
        pending.append(
            {
                "rank_index": 0,
                "candidate_id": candidate["candidate_id"],
                "rank_status": rank_status,
                "rank_score_microusd": rank_score_microusd,
                "pair_edge_feature_usd": feature_usd,
                "pair_edge_feature_bps": feature_bps,
                "reference_candidate_id": reference_candidate_id,
                "reference_venue_index": reference_venue_index,
                "reference_venue_id": reference_venue_id,
                "rank_tiebreak_key": (
                    f"{linkage_tiebreak}:{candidate['venue_index']:04d}:"
                    f"{candidate['side'].lower()}:{candidate['candidate_id']}"
                ),
                "feature_only": True,
                "admission_status": "HOLD",
                "admission_reason": EXPECTED_ADMISSION_REASON,
            }
        )
    pending.sort(
        key=lambda ranking: (
            0 if ranking["rank_status"] == "scored" else 1,
            -ranking["rank_score_microusd"],
            ranking["rank_tiebreak_key"],
        )
    )
    for idx, ranking in enumerate(pending, start=1):
        ranking["rank_index"] = idx
    return pending


def _best_same_side_reference(
    candidate: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    references = [
        reference
        for reference in candidates
        if reference["side"] == candidate["side"]
        and reference["venue_id"] != candidate["venue_id"]
        and reference["candidate_id"] != candidate["candidate_id"]
    ]
    if not references:
        return None
    if candidate["side"] == "Buy":
        return max(references, key=lambda reference: (reference["price"], reference["candidate_id"]))
    return min(references, key=lambda reference: (reference["price"], reference["candidate_id"]))


def _decision(
    *,
    scenario_id: str,
    now_ms: int,
    candidates: list[dict[str, Any]],
    baseline_plan_intent_count: int,
    baseline_mm_order_creating_intent_count: int,
) -> dict[str, Any]:
    return {
        "event_type": "V2_SHADOW_DECISION",
        "schema_version": 1,
        "telemetry_schema_version": 2,
        "scenario_id": scenario_id,
        "now_ms": now_ms,
        "decision_mode": "shadow",
        "admission_status": "HOLD",
        "admission_reason": EXPECTED_ADMISSION_REASON,
        "can_mutate_orders": False,
        "order_intent_output_count": 0,
        "baseline_plan_intent_count": baseline_plan_intent_count,
        "baseline_mm_order_creating_intent_count": baseline_mm_order_creating_intent_count,
        "pair_edge_is_admission": False,
        "pressure_complete_claim": False,
        "blocker_cleared": False,
        "require_phase51_gate": True,
        "pair_conditioned_admission_enabled": False,
        "fast_hedge_enabled": False,
        "order_intent_enabled": False,
        "ranking_schema_version": 1,
        "ranking_feature_only": True,
        "ranking_is_admission": False,
        "candidates": candidates,
        "candidate_rankings": _candidate_rankings(candidates),
        "pair_edges": [_pair_edge(candidates)],
    }


def build_scenarios() -> list[dict[str, Any]]:
    crossed_candidates: list[dict[str, Any]] = []
    bid_prices = {
        "extended": 100.0,
        "hyperliquid": 110.0,
        "aster": 102.0,
        "lighter": 103.0,
        "paradex": 104.0,
    }
    ask_prices = {
        "extended": 105.0,
        "hyperliquid": 104.0,
        "aster": 99.0,
        "lighter": 102.0,
        "paradex": 101.0,
    }
    for idx, venue_id in enumerate(VENUES):
        linkage = "present_redacted" if idx % 2 == 0 else "missing"
        crossed_candidates.append(
            _candidate(
                venue_index=idx,
                venue_id=venue_id,
                side="Buy",
                price=bid_prices[venue_id],
                size=0.01 + idx * 0.001,
                target_linkage_state=linkage,
            )
        )
        crossed_candidates.append(
            _candidate(
                venue_index=idx,
                venue_id=venue_id,
                side="Sell",
                price=ask_prices[venue_id],
                size=0.012 + idx * 0.001,
                target_linkage_state=linkage,
            )
        )

    bid_only_candidates = [
        _candidate(
            venue_index=idx,
            venue_id=venue_id,
            side="Buy",
            price=100.0 + idx,
            size=0.01,
            target_linkage_state="present_redacted",
        )
        for idx, venue_id in enumerate(VENUES)
    ]
    ask_only_candidates = [
        _candidate(
            venue_index=idx,
            venue_id=venue_id,
            side="Sell",
            price=101.0 + idx,
            size=0.01,
            target_linkage_state="missing",
        )
        for idx, venue_id in enumerate(VENUES)
    ]
    intent_fallback_candidates = [
        _intent_candidate(
            venue_index=3,
            venue_id="lighter",
            side="Buy",
            intent_index=0,
            price=99.5,
            size=0.01,
            target_linkage_state="present_redacted",
        ),
        _intent_candidate(
            venue_index=4,
            venue_id="paradex",
            side="Sell",
            intent_index=1,
            price=100.5,
            size=0.02,
            target_linkage_state="missing",
        ),
    ]

    return [
        _decision(
            scenario_id="all_five_crossed_pair_edge_feature",
            now_ms=1_779_220_000_000,
            candidates=crossed_candidates,
            baseline_plan_intent_count=len(crossed_candidates),
            baseline_mm_order_creating_intent_count=len(crossed_candidates),
        ),
        _decision(
            scenario_id="all_five_bid_only_missing_ask",
            now_ms=1_779_220_000_100,
            candidates=bid_only_candidates,
            baseline_plan_intent_count=len(bid_only_candidates),
            baseline_mm_order_creating_intent_count=len(bid_only_candidates),
        ),
        _decision(
            scenario_id="all_five_ask_only_missing_bid",
            now_ms=1_779_220_000_200,
            candidates=ask_only_candidates,
            baseline_plan_intent_count=len(ask_only_candidates),
            baseline_mm_order_creating_intent_count=len(ask_only_candidates),
        ),
        _decision(
            scenario_id="intent_fallback_place_replace_candidates",
            now_ms=1_779_220_000_300,
            candidates=intent_fallback_candidates,
            baseline_plan_intent_count=len(intent_fallback_candidates),
            baseline_mm_order_creating_intent_count=len(intent_fallback_candidates),
        ),
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_summary(path: Path, rows: list[dict[str, Any]], validation: validator.ValidationSummary) -> None:
    venue_counts = {venue: 0 for venue in VENUES}
    scenario_ids = []
    for row in rows:
        scenario_ids.append(row["scenario_id"])
        for candidate in row["candidates"]:
            venue_counts[candidate["venue_id"]] += 1
    summary = {
        "schema_version": 1,
        "artifact_type": "v2_shadow_scenario_matrix_summary",
        "scenario_count": len(rows),
        "scenario_ids": scenario_ids,
        "venues": list(VENUES),
        "venue_candidate_counts": venue_counts,
        "row_count": validation.row_count,
        "candidate_count_total": validation.candidate_count_total,
        "candidate_ranking_count_total": validation.candidate_ranking_count_total,
        "pair_edge_count_total": validation.pair_edge_count_total,
        "gate_status": "HOLD",
        "shadow_only": True,
        "blocker_cleared": False,
        "pressure_complete_claim": False,
        "live_orders_allowed": False,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_matrix(output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    decision_path = output_root / "v2_shadow_decisions.jsonl"
    summary_path = output_root / "scenario_summary.json"
    manifest_path = output_root / "manifest.json"
    rows = build_scenarios()
    _write_jsonl(decision_path, rows)
    validation = validator.validate_v2_shadow_decisions(decision_path)
    _write_summary(summary_path, rows, validation)
    manifest = validator.build_manifest(
        v2_shadow_decisions=decision_path,
        telemetry=None,
        summary_path=summary_path,
        validation=validation,
        artifact_root=output_root,
    )
    validator.write_manifest(manifest_path, manifest)
    return {
        "decision_path": decision_path,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        paths = generate_matrix(args.output_root)
    except (OSError, ValueError, validator.ContractViolation) as err:
        print(f"V2_SHADOW_SCENARIO_MATRIX_FAIL: {err}", file=sys.stderr)
        return 1
    print(f"V2_SHADOW_SCENARIO_MATRIX_WRITTEN output_root={args.output_root}")
    for label, path in paths.items():
        print(f"{label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
