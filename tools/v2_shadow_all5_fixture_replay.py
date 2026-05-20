#!/usr/bin/env python3
"""Generate deterministic V2 shadow evidence from sanitized all-five fixtures.

This is an offline validation artifact. It reads local sanitized top-of-book
fixtures, emits HOLD-only `V2_SHADOW_DECISION` and `V2_EV_EVALUATED` rows, and
validates them with `tools.v2_shadow_decision_validator`.
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
from tools import v2_shadow_scenario_matrix as matrix


DEFAULT_FIXTURE_PATHS = {
    "extended": Path("tests/fixtures/pnl_valid_all5/roadmap_b/extended/snapshot.json"),
    "hyperliquid": Path("tests/fixtures/pnl_valid_all5/hyperliquid/ws_l2_snapshot.json"),
    "aster": Path("tests/fixtures/pnl_valid_all5/roadmap_b/aster/snapshot.json"),
    "lighter": Path("tests/fixtures/pnl_valid_all5/lighter/ws_l2_snapshot.json"),
    "paradex": Path("tests/fixtures/pnl_valid_all5/roadmap_b/paradex/snapshot.json"),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"fixture must be a JSON object: {path}")
    return value


def _as_float(value: Any, *, field_name: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{field_name} must be numeric in {path}") from err
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive in {path}")
    return parsed


def _first_level(levels: Any, *, side: str, path: Path) -> tuple[float, float]:
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"{side} levels missing in {path}")
    first = levels[0]
    if not isinstance(first, list | tuple) or len(first) < 2:
        raise ValueError(f"{side} first level invalid in {path}")
    price = _as_float(first[0], field_name=f"{side}_price", path=path)
    size = _as_float(first[1], field_name=f"{side}_size", path=path)
    return price, size


def _extract_top_of_book(path: Path) -> tuple[float, float, float, float, int | None]:
    payload = _load_json(path)
    if "data" in payload and isinstance(payload["data"], dict):
        data = payload["data"]
        levels = data.get("levels")
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(f"hyperliquid levels missing in {path}")
        bids = levels[0]
        asks = levels[1]
        ts = data.get("time")
    else:
        bids = payload.get("bids")
        asks = payload.get("asks")
        ts = payload.get("timestamp_ms", payload.get("ts"))
    bid_price, bid_size = _first_level(bids, side="bid", path=path)
    ask_price, ask_size = _first_level(asks, side="ask", path=path)
    if bid_price >= ask_price:
        raise ValueError(f"fixture must preserve non-crossed top of book in {path}")
    timestamp_ms = int(ts) if isinstance(ts, int) and not isinstance(ts, bool) else None
    return bid_price, bid_size, ask_price, ask_size, timestamp_ms


def build_fixture_replay(repo_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    timestamps: list[int] = []
    for venue_index, venue_id in enumerate(matrix.VENUES):
        fixture_path = repo_root / DEFAULT_FIXTURE_PATHS[venue_id]
        bid_price, bid_size, ask_price, ask_size, timestamp_ms = _extract_top_of_book(fixture_path)
        if timestamp_ms is not None:
            timestamps.append(timestamp_ms)
        candidates.append(
            matrix._candidate(
                venue_index=venue_index,
                venue_id=venue_id,
                side="Buy",
                price=bid_price,
                size=bid_size,
                target_linkage_state="missing",
            )
        )
        candidates.append(
            matrix._candidate(
                venue_index=venue_index,
                venue_id=venue_id,
                side="Sell",
                price=ask_price,
                size=ask_size,
                target_linkage_state="missing",
            )
        )

    now_ms = max(timestamps) if timestamps else 1_779_220_010_000
    return [
        matrix._decision(
            scenario_id="all_five_fixture_replay_top_of_book",
            now_ms=now_ms,
            candidates=candidates,
            baseline_plan_intent_count=len(candidates),
            baseline_mm_order_creating_intent_count=len(candidates),
        )
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    repo_root: Path,
    decision_rows: list[dict[str, Any]],
    validation: validator.ValidationSummary,
) -> None:
    decision = decision_rows[0]
    venue_candidate_counts = {venue: 0 for venue in matrix.VENUES}
    for candidate in decision["candidates"]:
        venue_candidate_counts[candidate["venue_id"]] += 1
    fixture_paths = {
        venue: str((repo_root / relative_path).resolve().relative_to(repo_root.resolve()))
        for venue, relative_path in DEFAULT_FIXTURE_PATHS.items()
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "v2_shadow_all5_fixture_replay_summary",
        "scenario_id": decision["scenario_id"],
        "fixtures_are_sanitized_local_inputs": True,
        "fixture_paths": fixture_paths,
        "venues": list(matrix.VENUES),
        "venue_candidate_counts": venue_candidate_counts,
        "row_count": validation.row_count,
        "shadow_decision_row_count": validation.shadow_decision_row_count,
        "ev_evaluation_count_total": validation.ev_evaluation_count_total,
        "candidate_count_total": validation.candidate_count_total,
        "candidate_ranking_count_total": validation.candidate_ranking_count_total,
        "pair_edge_count_total": validation.pair_edge_count_total,
        "target_linkage_state": "missing",
        "gate_status": "HOLD",
        "shadow_only": True,
        "blocker_cleared": False,
        "pressure_complete_claim": False,
        "live_orders_allowed": False,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_fixture_replay(output_root: Path, *, repo_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    decision_path = output_root / "v2_shadow_fixture_replay.jsonl"
    summary_path = output_root / "fixture_replay_summary.json"
    manifest_path = output_root / "manifest.json"
    decision_rows = build_fixture_replay(repo_root)
    _write_jsonl(decision_path, matrix._evidence_rows(decision_rows))
    validation = validator.validate_v2_shadow_decisions(
        decision_path,
        require_ev_evaluations=True,
    )
    _write_summary(
        summary_path,
        repo_root=repo_root,
        decision_rows=decision_rows,
        validation=validation,
    )
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
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        paths = generate_fixture_replay(args.output_root, repo_root=args.repo_root)
    except (OSError, ValueError, validator.ContractViolation) as err:
        print(f"V2_SHADOW_ALL5_FIXTURE_REPLAY_FAIL: {err}", file=sys.stderr)
        return 1
    print(f"V2_SHADOW_ALL5_FIXTURE_REPLAY_WRITTEN output_root={args.output_root}")
    for label, path in paths.items():
        print(f"{label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
