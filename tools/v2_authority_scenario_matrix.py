#!/usr/bin/env python3
"""Generate deterministic V2 paper-admission authority fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _gate_state(**overrides: bool) -> dict[str, bool]:
    state = {
        "enabled": True,
        "decision_mode_is_paper_admission": True,
        "execution_mode_is_paper": True,
        "pair_edge_enabled": True,
        "pair_conditioned_admission_enabled": True,
        "order_intent_enabled": True,
        "fast_hedge_disabled": True,
        "require_phase51_gate": True,
    }
    state.update(overrides)
    return state


def _admitted_row() -> dict[str, Any]:
    admitted_candidates = [
        {
            "candidate_id": f"v2_shadow_intent_v1:{idx}:{venue}:buy:{idx}",
            "venue_index": idx,
            "venue_id": venue,
            "side": "Buy",
            "rank_index": idx + 1,
            "rank_score_microusd": 2_000_000 - idx,
            "pair_edge_feature_usd": 2.0,
            "pair_edge_feature_bps": 2.5,
            "reference_candidate_id": "v2_shadow_intent_v1:1:hyperliquid:buy:1",
        }
        for idx, venue in enumerate(["extended", "hyperliquid", "aster", "lighter", "paradex"])
    ]
    return {
        "scenario_id": "all_five_positive_pair_edge",
        "event_type": "V2_ADMISSION_DECISION",
        "schema_version": 1,
        "telemetry_schema_version": 2,
        "now_ms": 1_779_226_270_000,
        "decision_mode": "paper_admission",
        "execution_mode": "paper",
        "authority_scope": "paper_only",
        "admission_status": "ADMITTED",
        "admission_reason": "paper_positive_pair_edge_ranked_admission",
        "can_filter_existing_intents": True,
        "can_create_new_intents": False,
        "can_mutate_live_orders": False,
        "order_intent_output_count": len(admitted_candidates),
        "baseline_plan_intent_count": 10,
        "baseline_mm_order_creating_intent_count": 10,
        "suppressed_mm_order_creating_intent_count": 10 - len(admitted_candidates),
        "pair_edge_is_admission": True,
        "pressure_complete_claim": False,
        "blocker_cleared": False,
        "gate_state": _gate_state(),
        "ranking_schema_version": 1,
        "ranking_feature_only": False,
        "ranking_is_admission": True,
        "pair_edges": [
            {
                "snapshot_id": "v2_pair_edge_v1:v2_shadow_intent_v1:0:extended:buy:0:v2_shadow_intent_v1:1:lighter:sell:1",
                "bid_candidate_id": "v2_shadow_intent_v1:0:extended:buy:0",
                "ask_candidate_id": "v2_shadow_intent_v1:1:lighter:sell:1",
                "edge_usd": 2.0,
                "edge_bps": 2.5,
                "feature_only": False,
                "invalid_reason": None,
            }
        ],
        "admitted_candidates": admitted_candidates,
    }


def build_matrix() -> list[dict[str, Any]]:
    rows = [_admitted_row()]
    for field in [
        "enabled",
        "execution_mode_is_paper",
        "pair_edge_enabled",
        "pair_conditioned_admission_enabled",
        "order_intent_enabled",
        "fast_hedge_disabled",
        "require_phase51_gate",
    ]:
        row = _admitted_row()
        row["scenario_id"] = f"missing_gate_{field}"
        row["admission_status"] = "HOLD"
        row["admission_reason"] = "paper_admission_gate_not_satisfied"
        row["order_intent_output_count"] = 0
        row["suppressed_mm_order_creating_intent_count"] = 0
        row["pair_edge_is_admission"] = False
        row["ranking_is_admission"] = False
        row["can_filter_existing_intents"] = False
        row["gate_state"] = _gate_state(**{field: False})
        row["admitted_candidates"] = []
        rows.append(row)
    row = _admitted_row()
    row["scenario_id"] = "negative_pair_edge"
    row["admission_status"] = "HOLD"
    row["admission_reason"] = "no_positive_pair_edge"
    row["order_intent_output_count"] = 0
    row["suppressed_mm_order_creating_intent_count"] = row["baseline_mm_order_creating_intent_count"]
    row["pair_edge_is_admission"] = False
    row["ranking_is_admission"] = False
    row["pair_edges"][0]["edge_usd"] = -1.0
    row["admitted_candidates"] = []
    rows.append(row)
    for scenario_id, invalid_reason, bid, ask in [
        ("missing_bid", "missing_bid", None, "v2_shadow_intent_v1:1:lighter:sell:1"),
        ("missing_ask", "missing_ask", "v2_shadow_intent_v1:0:extended:buy:0", None),
    ]:
        row = _admitted_row()
        row["scenario_id"] = scenario_id
        row["admission_status"] = "HOLD"
        row["admission_reason"] = invalid_reason
        row["order_intent_output_count"] = 0
        row["suppressed_mm_order_creating_intent_count"] = row["baseline_mm_order_creating_intent_count"]
        row["pair_edge_is_admission"] = False
        row["ranking_is_admission"] = False
        row["pair_edges"] = [
            {
                "snapshot_id": f"v2_pair_edge_v1:{scenario_id}",
                "bid_candidate_id": bid,
                "ask_candidate_id": ask,
                "edge_usd": None,
                "edge_bps": None,
                "feature_only": False,
                "invalid_reason": invalid_reason,
            }
        ]
        row["admitted_candidates"] = []
        rows.append(row)
    row = _admitted_row()
    row["scenario_id"] = "positive_pair_edge_no_admitted_candidates"
    row["admission_status"] = "HOLD"
    row["admission_reason"] = "no_admitted_candidates"
    row["order_intent_output_count"] = 0
    row["suppressed_mm_order_creating_intent_count"] = row["baseline_mm_order_creating_intent_count"]
    row["pair_edge_is_admission"] = False
    row["ranking_is_admission"] = False
    row["admitted_candidates"] = []
    rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    decision_path = args.output_root / "v2_authority_decisions.jsonl"
    rows = build_matrix()
    decision_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "artifact_type": "v2_authority_scenario_matrix_summary",
        "row_count": len(rows),
        "decision_path": decision_path.name,
    }
    (args.output_root / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"V2_AUTHORITY_SCENARIO_MATRIX_WRITTEN output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
