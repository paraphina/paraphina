#!/usr/bin/env python3
"""Build the Phase 5.1e lifecycle/native-truth audit pack.

This is an offline evidence gate. It diagnoses whether Phase 5.1c P_fill
censoring is explained by order identity aliasing, lifecycle parser gaps, or
missing venue-native truth. It does not rewrite labels, train models, submit
orders, approve EV admission, or make financial claims.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51e_lifecycle_truth_audit"

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

PLACE_ACTION = "place"
DIRECT_TERMINAL_ACTIONS = {"cancel", "expire", "expired", "reject", "rejected"}
CHAIN_TERMINAL_ACTIONS = {"replace"}
GLOBAL_TERMINAL_ACTIONS = {"cancel_all"}
TERMINAL_ACTIONS = DIRECT_TERMINAL_ACTIONS | CHAIN_TERMINAL_ACTIONS | GLOBAL_TERMINAL_ACTIONS


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, node: str) -> None:
        self.parent.setdefault(node, node)

    def find(self, node: str) -> str:
        self.add(node)
        parent = self.parent[node]
        if parent != node:
            parent = self.find(parent)
            self.parent[node] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


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


def _resolve_path(path_text: Any, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return path
    fallback = base_dir / str(path_text)
    return fallback if fallback.exists() else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_side(value: Any) -> str:
    side = str(value or "").strip().lower()
    if side in {"bid", "buy"}:
        return "BID"
    if side in {"ask", "sell"}:
        return "ASK"
    return "UNKNOWN"


def _check_unsafe(record: dict[str, Any], path: Path, *, label: str) -> None:
    for flag in UNSAFE_TRUE_FLAGS:
        if record.get(flag) is True:
            raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _load_summary(path: Path, *, expected_file: str) -> dict[str, Any]:
    summary_path = path / expected_file
    summary = _load_json(summary_path)
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    if summary.get("gate_status") != "HOLD":
        raise ValueError(f"{summary_path} must have gate_status=HOLD")
    _check_unsafe(summary, summary_path, label="summary")
    return summary


def _node_id(source_sha: str, label: dict[str, Any]) -> str:
    return f"{source_sha}:order_label_seq:{label.get('label_seq')}"


def _alias_keys(source_sha: str, label: dict[str, Any]) -> list[tuple[str, str]]:
    venue = str(label.get("venue_id") or "UNKNOWN").lower()
    keys: list[tuple[str, str]] = []
    for field in ("order_id_hash", "client_order_id_hash", "decision_id"):
        value = label.get(field)
        if value:
            keys.append((f"{source_sha}:{venue}:{field}", str(value)))
    return keys


def _join_alias_keys(source_sha: str, label: dict[str, Any]) -> list[tuple[str, str]]:
    venue = str(label.get("venue_id") or "UNKNOWN").lower()
    keys: list[tuple[str, str]] = []
    for field in ("order_id_hash", "client_order_id_hash", "order_decision_id", "fill_decision_id"):
        value = label.get(field)
        if not value:
            continue
        kind = "decision_id" if field.endswith("decision_id") else field
        keys.append((f"{source_sha}:{venue}:{kind}", str(value)))
    return keys


def _action(label: dict[str, Any]) -> str:
    return str(label.get("action") or "UNKNOWN").strip().lower()


def _status(label: dict[str, Any]) -> str:
    return str(label.get("status") or "UNKNOWN").strip().lower()


def _load_lifecycle_graph(label_lake_run: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str], str]]:
    summary = _load_summary(label_lake_run, expected_file="label_lake_summary.json")
    source_sha = str(summary.get("source_telemetry_sha256") or "")
    if not source_sha:
        raise ValueError(f"{label_lake_run} missing source_telemetry_sha256")
    uf = UnionFind()
    alias_to_node: dict[tuple[str, str], str] = {}
    lifecycle_by_node: dict[str, dict[str, Any]] = {}
    place_by_seq: dict[str, dict[str, Any]] = {}
    labels_path = label_lake_run / "labels.jsonl"
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "ORDER_LIFECYCLE_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        node = _node_id(source_sha, label)
        uf.add(node)
        lifecycle_by_node[node] = label
        if _action(label) == PLACE_ACTION:
            place_by_seq[str(label.get("label_seq"))] = label
        for alias in _alias_keys(source_sha, label):
            previous = alias_to_node.get(alias)
            if previous is None:
                alias_to_node[alias] = node
            else:
                uf.union(previous, node)
    group_by_node = {node: uf.find(node) for node in lifecycle_by_node}
    return summary, lifecycle_by_node, place_by_seq, {alias: uf.find(node) for alias, node in alias_to_node.items()} | {
        ("node", node): group for node, group in group_by_node.items()
    }


def _resolve_group(source_sha: str, label: dict[str, Any], group_index: dict[tuple[str, str], str]) -> tuple[str | None, str]:
    label_seq = label.get("order_label_seq") or label.get("label_seq")
    if label_seq is not None:
        node = f"{source_sha}:order_label_seq:{label_seq}"
        group = group_index.get(("node", node))
        if group:
            return group, "order_label_seq"
    for alias in _join_alias_keys(source_sha, label) or _alias_keys(source_sha, label):
        group = group_index.get(alias)
        if group:
            return group, alias[0].split(":")[-1]
    return None, "unmatched"


def _group_facts(lifecycle_by_node: dict[str, dict[str, Any]], group_index: dict[tuple[str, str], str]) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    for node, label in lifecycle_by_node.items():
        group = group_index[("node", node)]
        fact = facts.setdefault(
            group,
            {
                "group_id": group,
                "venue_id": label.get("venue_id"),
                "actions": {},
                "statuses": {},
                "place_label_seqs": [],
                "place_intent_count": 0,
                "place_ack_count": 0,
                "direct_terminal_count": 0,
                "replace_terminal_count": 0,
                "cancel_all_count": 0,
                "first_source_line": None,
                "last_source_line": None,
            },
        )
        action = _action(label)
        status = _status(label)
        fact["actions"][action] = fact["actions"].get(action, 0) + 1
        fact["statuses"][status] = fact["statuses"].get(status, 0) + 1
        line = _safe_int(label.get("source_line"))
        if line is not None:
            fact["first_source_line"] = line if fact["first_source_line"] is None else min(fact["first_source_line"], line)
            fact["last_source_line"] = line if fact["last_source_line"] is None else max(fact["last_source_line"], line)
        if action == PLACE_ACTION:
            fact["place_label_seqs"].append(label.get("label_seq"))
            if status == "intent":
                fact["place_intent_count"] += 1
            elif status == "ack":
                fact["place_ack_count"] += 1
        elif action in DIRECT_TERMINAL_ACTIONS:
            fact["direct_terminal_count"] += 1
        elif action in CHAIN_TERMINAL_ACTIONS:
            fact["replace_terminal_count"] += 1
        elif action in GLOBAL_TERMINAL_ACTIONS:
            fact["cancel_all_count"] += 1
    for fact in facts.values():
        fact["place_count"] = len(fact["place_label_seqs"])
        fact["terminal_event_count"] = (
            fact["direct_terminal_count"] + fact["replace_terminal_count"] + fact["cancel_all_count"]
        )
        fact["has_terminal_event"] = fact["terminal_event_count"] > 0
        fact["has_duplicate_place_aliases"] = fact["place_count"] > 1
    return facts


def _load_pfill_runs(pfill_outcome_runs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_path in pfill_outcome_runs:
        summary = _load_summary(run_path, expected_file="pfill_outcome_summary.json")
        labels_path = run_path / "pfill_order_labels.jsonl"
        run_labels = []
        for _, label in _iter_jsonl(labels_path):
            if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
                continue
            _check_unsafe(label, labels_path, label="label")
            run_labels.append(label)
        expected = {
            "order_label_count": int(summary.get("order_label_count") or 0),
            "filled_count": int(summary.get("filled_count") or 0),
            "not_filled_count": int(summary.get("not_filled_count") or 0),
            "censored_count": int(summary.get("censored_count") or 0),
        }
        actual = {
            "order_label_count": len(run_labels),
            "filled_count": sum(1 for row in run_labels if row.get("outcome_status") == "OBSERVED_FILLED"),
            "not_filled_count": sum(1 for row in run_labels if row.get("outcome_status") == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL"),
            "censored_count": sum(1 for row in run_labels if row.get("outcome_status") == "CENSORED_OR_UNOBSERVED"),
        }
        if actual != expected:
            raise ValueError(f"{run_path} summary counts do not reconcile: {actual} != {expected}")
        summaries.append({
            "run_path": str(run_path),
            "run_id": summary.get("run_id"),
            "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
            "label_lake_run": summary.get("label_lake_run"),
            "join_holdout_run": summary.get("join_holdout_run"),
            "pfill_outcome_summary_sha256": _sha256_file(run_path / "pfill_outcome_summary.json"),
            "pfill_order_labels_sha256": _sha256_file(labels_path),
            **expected,
        })
        for label in run_labels:
            enriched = dict(label)
            enriched["_pfill_run_path"] = str(run_path)
            enriched["_source_telemetry_sha256"] = summary.get("source_telemetry_sha256")
            enriched["_label_lake_run"] = summary.get("label_lake_run")
            enriched["_join_holdout_run"] = summary.get("join_holdout_run")
            labels.append(enriched)
    return summaries, labels


def _load_join_fills(join_run: Path, source_sha: str, group_index: dict[tuple[str, str], str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, Any]]:
    summary = _load_summary(join_run, expected_file="join_holdout_summary.json")
    if summary.get("source_telemetry_sha256") != source_sha:
        raise ValueError(f"{join_run} source_telemetry_sha256 mismatch")
    fills_by_group: dict[str, list[dict[str, Any]]] = {}
    unresolved: dict[str, int] = {}
    labels_path = join_run / "joined_labels.jsonl"
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "DETERMINISTIC_JOIN_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        group, reason = _resolve_group(source_sha, label, group_index)
        if group:
            fills_by_group.setdefault(group, []).append(label)
        else:
            unresolved[reason] = unresolved.get(reason, 0) + 1
    return fills_by_group, unresolved, {
        "run_path": str(join_run),
        "run_id": summary.get("run_id"),
        "join_holdout_summary_sha256": _sha256_file(join_run / "join_holdout_summary.json"),
        "joined_labels_sha256": _sha256_file(labels_path),
    }


def _load_cancel_all_by_venue(lifecycle_by_node: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for label in lifecycle_by_node.values():
        if _action(label) != "cancel_all":
            continue
        venue = str(label.get("venue_id") or "UNKNOWN").lower()
        line = _safe_int(label.get("source_line"))
        if line is not None:
            result.setdefault(venue, []).append(line)
    return result


def _has_later_cancel_all(label: dict[str, Any], cancel_all_by_venue: dict[str, list[int]]) -> bool:
    venue = str(label.get("venue_id") or "UNKNOWN").lower()
    line = _safe_int(label.get("order_source_line"))
    if line is None:
        return False
    return any(cancel_line >= line for cancel_line in cancel_all_by_venue.get(venue, []))


def _current_status(label: dict[str, Any]) -> str:
    status = str(label.get("outcome_status") or "")
    if status == "OBSERVED_FILLED":
        return "CURRENT_OBSERVED_FILLED"
    if status == "OBSERVED_NOT_FILLED_TO_TERMINAL_CANCEL":
        return "CURRENT_OBSERVED_NOT_FILLED"
    if status == "CENSORED_OR_UNOBSERVED":
        return "CURRENT_CENSORED"
    return "CURRENT_UNKNOWN"


def _audit_status(
    label: dict[str, Any],
    *,
    group_fact: dict[str, Any] | None,
    group_fill_count: int,
    group_observed_statuses: set[str],
    later_cancel_all: bool,
) -> tuple[str, str]:
    current = _current_status(label)
    if current == "CURRENT_OBSERVED_FILLED":
        return "STAYS_FILLED", "already observed filled in Phase 5.1c P_fill outcome"
    if current == "CURRENT_OBSERVED_NOT_FILLED":
        return "STAYS_NOT_FILLED", "already observed terminal not-filled in Phase 5.1c P_fill outcome"
    if current != "CURRENT_CENSORED":
        return "PARSER_GAP", "unexpected current P_fill outcome status"
    if group_fill_count > 0 or "OBSERVED_FILLED" in group_observed_statuses:
        return "CENSORED_TO_CANONICAL_FILLED_REVIEW", "canonical lifecycle group has fill evidence"
    if group_fact and group_fact.get("direct_terminal_count", 0) > 0:
        return "CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW", "canonical lifecycle group has direct terminal event"
    if group_fact and group_fact.get("replace_terminal_count", 0) > 0:
        return "CENSORED_TO_REPLACE_CHAIN_REVIEW", "canonical lifecycle group has replace-chain terminal evidence"
    if group_fact and group_fact.get("has_duplicate_place_aliases"):
        return "DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW", "place intent/ack aliases collapse to one lifecycle group"
    if later_cancel_all:
        return "CANCEL_ALL_SCOPE_REVIEW", "later venue-level cancel_all may terminate order but lacks exact order id"
    if group_fact is None:
        return "ORDER_IDENTITY_GAP_REVIEW", "P_fill label cannot be resolved to lifecycle graph"
    return "REMAINS_NO_TERMINAL_EVENT_WITH_SUFFICIENT_WINDOW", "no canonical fill or terminal evidence found"


def _load_lighter_gap_labels(lighter_gap_run: Path | None, source_sha: str | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if lighter_gap_run is None:
        return [], None
    summary = _load_summary(lighter_gap_run, expected_file="lighter_attribution_gap_summary.json")
    if source_sha and summary.get("source_telemetry_sha256") != source_sha:
        raise ValueError(f"{lighter_gap_run} source_telemetry_sha256 mismatch")
    labels_path = lighter_gap_run / "lighter_attribution_gap_labels.jsonl"
    labels: list[dict[str, Any]] = []
    for _, label in _iter_jsonl(labels_path):
        if label.get("label_type") != "LIGHTER_ATTRIBUTION_GAP_AUDIT_LABEL":
            continue
        _check_unsafe(label, labels_path, label="label")
        labels.append(label)
    return labels, {
        "run_path": str(lighter_gap_run),
        "run_id": summary.get("run_id"),
        "source_telemetry_sha256": summary.get("source_telemetry_sha256"),
        "observed_run": summary.get("observed_run"),
        "lighter_trade_backfill_run": summary.get("lighter_trade_backfill_run"),
        "lighter_attribution_gap_summary_sha256": _sha256_file(lighter_gap_run / "lighter_attribution_gap_summary.json"),
        "lighter_attribution_gap_labels_sha256": _sha256_file(labels_path),
        "lighter_fill_count": summary.get("lighter_fill_count"),
        "observed_role_counts": summary.get("observed_role_counts"),
        "gap_reason_counts": summary.get("gap_reason_counts"),
    }


def _iter_json_stream(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            source = line.strip()
            if not source:
                continue
            pos = 0
            while pos < len(source):
                try:
                    record, end = decoder.raw_decode(source, pos)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}:{pos + 1}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_no}")
                yield line_no, record
                pos = end
                while pos < len(source) and source[pos].isspace():
                    pos += 1


def _string_ids(*values: Any) -> list[str]:
    ids: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            ids.append(text)
    return ids


def _hash_or_none(value: Any) -> str | None:
    return _stable_hash(value) if value not in (None, "") else None


def _extract_items(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        value = lowered.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _load_lighter_backfill_trades(trade_backfill_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int | None]:
    summary = _load_summary(trade_backfill_run, expected_file="lighter_trade_backfill_summary.json")
    trades_path = Path(str(summary.get("trades_path") or ""))
    if not trades_path.is_absolute():
        trades_path = ROOT / trades_path
    payload = _load_json(trades_path)
    trades = _extract_items(payload, {"trades", "trade_history", "tradehistory"})
    account_index = _safe_int(summary.get("account_index"))
    if account_index is None and isinstance(payload, dict):
        account_index = _safe_int(payload.get("account_index"))
    return summary, trades, account_index


def _trade_timestamp_ms(trade: dict[str, Any]) -> int | None:
    return _safe_int(trade.get("timestamp") or trade.get("time") or trade.get("created_at"))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trade_side_ids(trade: dict[str, Any], side: str) -> list[str]:
    if side == "ASK":
        return _string_ids(trade.get("ask_id"), trade.get("ask_id_str"), trade.get("ask_client_id"), trade.get("ask_client_id_str"))
    if side == "BID":
        return _string_ids(trade.get("bid_id"), trade.get("bid_id_str"), trade.get("bid_client_id"), trade.get("bid_client_id_str"))
    return []


def _trade_all_ids(trade: dict[str, Any]) -> list[str]:
    return _string_ids(
        trade.get("ask_id"),
        trade.get("ask_id_str"),
        trade.get("ask_client_id"),
        trade.get("ask_client_id_str"),
        trade.get("bid_id"),
        trade.get("bid_id_str"),
        trade.get("bid_client_id"),
        trade.get("bid_client_id_str"),
    )


def _native_role_for_account(trade: dict[str, Any], account_index: int | None) -> tuple[str | None, str | None]:
    if account_index is None:
        return None, "NATIVE_ACCOUNT_INDEX_MISSING"
    is_maker_ask = trade.get("is_maker_ask")
    if is_maker_ask is None:
        is_maker_ask = trade.get("isMakerAsk")
    if not isinstance(is_maker_ask, bool):
        return None, "NATIVE_ACCOUNT_ROLE_UNDERIVABLE"
    ask_account = _safe_int(trade.get("ask_account_id") or trade.get("askAccountId"))
    bid_account = _safe_int(trade.get("bid_account_id") or trade.get("bidAccountId"))
    if account_index == ask_account:
        return ("MAKER" if is_maker_ask else "TAKER"), None
    if account_index == bid_account:
        return ("TAKER" if is_maker_ask else "MAKER"), None
    return None, "NATIVE_ACCOUNT_ROLE_UNDERIVABLE"


def _raw_fill_id(line_no: int, source_t: Any, fill_index: int, fill: dict[str, Any]) -> str:
    return _stable_hash([
        line_no,
        source_t,
        fill_index,
        fill.get("venue_id"),
        fill.get("order_id"),
        fill.get("client_order_id"),
        fill.get("fill_time_ms"),
        fill.get("side"),
        fill.get("price"),
        fill.get("size"),
    ])


def _match_native_trade(
    fill: dict[str, Any],
    trades: list[dict[str, Any]],
    trades_by_raw_id: dict[str, dict[str, Any]],
    *,
    time_tolerance_ms: int,
    price_tolerance: float,
    size_tolerance: float,
) -> tuple[str, str, dict[str, Any] | None]:
    raw_ids = _string_ids(fill.get("order_id"), fill.get("client_order_id"))
    for raw_id in raw_ids:
        trade = trades_by_raw_id.get(raw_id)
        if trade is not None:
            return "MATCHED_NATIVE_ID", "raw telemetry order/client id matched native trade side id", trade
    if not raw_ids:
        return "RAW_TELEMETRY_ID_MISSING", "raw telemetry fill has no order_id or client_order_id", None
    fill_ts = _safe_int(fill.get("fill_time_ms"))
    fill_price = _as_float(fill.get("price"))
    fill_size = _as_float(fill.get("size"))
    if fill_ts is None or fill_price is None or fill_size is None:
        return "PARSER_GAP", "fill missing timestamp, price, or size", None
    candidates = []
    for trade in trades:
        ts = _trade_timestamp_ms(trade)
        price = _as_float(trade.get("price"))
        size = _as_float(trade.get("size"))
        if ts is None or price is None or size is None:
            continue
        if (
            abs(ts - fill_ts) <= time_tolerance_ms
            and abs(price - fill_price) <= price_tolerance
            and abs(size - fill_size) <= size_tolerance
        ):
            candidates.append(trade)
    if len(candidates) > 1:
        return "TIME_PRICE_SIZE_AMBIGUOUS", "multiple native trades match time/price/size", None
    if len(candidates) == 1:
        candidate = candidates[0]
        side = _canonical_side(fill.get("side"))
        side_ids = _trade_side_ids(candidate, side)
        if side_ids and not any(raw_id in side_ids for raw_id in raw_ids):
            return "ORDER_ID_MISMATCH", "time/price/size candidate has different native side order id", candidate
        all_ids = _trade_all_ids(candidate)
        if all_ids:
            return "CLIENT_ID_MISMATCH", "time/price/size candidate exists but raw telemetry id does not match native ids", candidate
        return "PARSER_GAP", "time/price/size candidate lacks usable native ids", candidate
    return "NATIVE_WINDOW_COVERED_NO_MATCH", "native backfill window covers fill but no native trade matches", None


def _raw_lighter_native_truth_records(
    *,
    observed_run: Path,
    lighter_trade_backfill_run: Path,
    run_id: str,
    timestamp_ns: int,
    time_tolerance_ms: int,
    price_tolerance: float,
    size_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_summary = _load_summary(observed_run, expected_file="observed_label_summary.json")
    source_telemetry = Path(str(observed_summary.get("source_telemetry") or ""))
    if not source_telemetry.is_absolute():
        source_telemetry = ROOT / source_telemetry
    if not source_telemetry.exists():
        raise ValueError(f"observed source telemetry does not exist: {source_telemetry}")
    trade_summary, trades, account_index = _load_lighter_backfill_trades(lighter_trade_backfill_run)
    trades_by_raw_id: dict[str, dict[str, Any]] = {}
    for trade in trades:
        for raw_id in _trade_all_ids(trade):
            trades_by_raw_id.setdefault(raw_id, trade)
    trade_timestamps = [ts for ts in (_trade_timestamp_ms(trade) for trade in trades) if ts is not None]
    trade_min_ts = min(trade_timestamps) if trade_timestamps else None
    trade_max_ts = max(trade_timestamps) if trade_timestamps else None
    records: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for line_no, source_record in _iter_json_stream(source_telemetry):
        source_t = source_record.get("t")
        for fill_index, fill in enumerate(source_record.get("fills") or []):
            if not isinstance(fill, dict):
                continue
            if str(fill.get("venue_id") or "").lower() != "lighter":
                continue
            fill_ts = _safe_int(fill.get("fill_time_ms"))
            if trade_min_ts is not None and trade_max_ts is not None and fill_ts is not None:
                outside_window = fill_ts < trade_min_ts - time_tolerance_ms or fill_ts > trade_max_ts + time_tolerance_ms
            else:
                outside_window = False
            if outside_window:
                match_status = "NATIVE_BACKFILL_INCOMPLETE_FOR_FILL"
                match_reason = "fill timestamp outside native trade backfill window"
                native_trade = None
            else:
                match_status, match_reason, native_trade = _match_native_trade(
                    fill,
                    trades,
                    trades_by_raw_id,
                    time_tolerance_ms=time_tolerance_ms,
                    price_tolerance=price_tolerance,
                    size_tolerance=size_tolerance,
                )
            native_role, role_error = _native_role_for_account(native_trade, account_index) if native_trade else (None, None)
            if role_error and match_status == "MATCHED_NATIVE_ID":
                match_status = role_error
                match_reason = role_error.lower()
            reason_counts[match_status] = reason_counts.get(match_status, 0) + 1
            role_counts[native_role or "UNKNOWN"] = role_counts.get(native_role or "UNKNOWN", 0) + 1
            records.append({
                "schema_version": 1,
                "label_type": "PHASE51E_LIGHTER_RAW_NATIVE_TRUTH_LABEL",
                "run_id": run_id,
                "baseline_commit": BASELINE_COMMIT,
                "timestamp_local_ns": timestamp_ns + len(records) + 1,
                "source_telemetry_sha256": observed_summary.get("source_telemetry_sha256"),
                "source_line": line_no,
                "source_t": source_t,
                "source_fill_index": fill_index,
                "fill_id": _raw_fill_id(line_no, source_t, fill_index, fill),
                "fill_time_ms": fill.get("fill_time_ms"),
                "venue_id": "lighter",
                "side": _canonical_side(fill.get("side")),
                "price": fill.get("price"),
                "size": fill.get("size"),
                "purpose": fill.get("purpose"),
                "decision_id_hash": _hash_or_none(fill.get("decision_id")),
                "telemetry_order_id_hash": _hash_or_none(fill.get("order_id")),
                "telemetry_client_order_id_hash": _hash_or_none(fill.get("client_order_id")),
                "telemetry_order_id_present": fill.get("order_id") not in (None, ""),
                "telemetry_client_order_id_present": fill.get("client_order_id") not in (None, ""),
                "native_trade_match_status": match_status,
                "native_trade_match_reason": match_reason,
                "native_trade_id_str_hash": _hash_or_none(
                    native_trade.get("trade_id_str") or native_trade.get("trade_id") if native_trade else None
                ),
                "native_trade_timestamp_ms": _trade_timestamp_ms(native_trade) if native_trade else None,
                "native_transaction_time": native_trade.get("transaction_time") if native_trade else None,
                "native_role": native_role,
                "native_is_maker_ask": native_trade.get("is_maker_ask") if native_trade else None,
                "native_ask_id_hash": _hash_or_none(native_trade.get("ask_id") if native_trade else None),
                "native_ask_client_id_hash": _hash_or_none(native_trade.get("ask_client_id") if native_trade else None),
                "native_bid_id_hash": _hash_or_none(native_trade.get("bid_id") if native_trade else None),
                "native_bid_client_id_hash": _hash_or_none(native_trade.get("bid_client_id") if native_trade else None),
                "time_delta_ms": (
                    fill_ts - _trade_timestamp_ms(native_trade)
                    if fill_ts is not None and native_trade and _trade_timestamp_ms(native_trade) is not None
                    else None
                ),
                "price_delta": (
                    _as_float(fill.get("price")) - _as_float(native_trade.get("price"))
                    if native_trade and _as_float(fill.get("price")) is not None and _as_float(native_trade.get("price")) is not None
                    else None
                ),
                "size_delta": (
                    _as_float(fill.get("size")) - _as_float(native_trade.get("size"))
                    if native_trade and _as_float(fill.get("size")) is not None and _as_float(native_trade.get("size")) is not None
                    else None
                ),
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
            })
    return records, {
        "observed_run": str(observed_run),
        "lighter_trade_backfill_run": str(lighter_trade_backfill_run),
        "source_telemetry": str(source_telemetry),
        "source_telemetry_sha256": observed_summary.get("source_telemetry_sha256"),
        "observed_summary_sha256": _sha256_file(observed_run / "observed_label_summary.json"),
        "lighter_trade_backfill_summary_sha256": _sha256_file(lighter_trade_backfill_run / "lighter_trade_backfill_summary.json"),
        "native_account_index_present": account_index is not None,
        "native_trade_count": len(trades),
        "native_trade_timestamp_min_ms": trade_min_ts,
        "native_trade_timestamp_max_ms": trade_max_ts,
        "raw_lighter_fill_count": len(records),
        "raw_native_match_status_counts": dict(sorted(reason_counts.items())),
        "raw_native_role_counts": dict(sorted(role_counts.items())),
    }


def build_lifecycle_truth_audit(
    *,
    pfill_outcome_runs: list[Path],
    label_lake_run: Path | None,
    join_holdout_run: Path | None,
    lighter_attribution_gap_run: Path | None,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
) -> Path:
    if not pfill_outcome_runs:
        raise ValueError("at least one --pfill-outcome-run is required")
    pfill_summaries, pfill_labels = _load_pfill_runs(pfill_outcome_runs)
    run_id = run_id or f"PHASE51E-LIFECYCLE-TRUTH-AUDIT-{_utc_stamp()}"
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    source_summaries: dict[str, dict[str, Any]] = {}
    for summary in pfill_summaries:
        source_sha = str(summary.get("source_telemetry_sha256") or "")
        if not source_sha:
            raise ValueError("pfill outcome summary missing source_telemetry_sha256")
        source_summaries.setdefault(source_sha, summary)
    if (label_lake_run is not None or join_holdout_run is not None) and len(source_summaries) > 1:
        raise ValueError("explicit --label-lake-run/--join-holdout-run may only be used with one source telemetry SHA")

    labels_by_source: dict[str, list[dict[str, Any]]] = {}
    for label in pfill_labels:
        source_sha = str(label.get("_source_telemetry_sha256") or label.get("source_telemetry_sha256") or "")
        if not source_sha:
            raise ValueError("pfill label missing source_telemetry_sha256")
        labels_by_source.setdefault(source_sha, []).append(label)

    source_input_summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    unresolved_join_reason_counts_total: dict[str, int] = {}
    current_counts = {
        "order_label_count": 0,
        "current_filled_count": 0,
        "current_not_filled_count": 0,
        "current_censored_count": 0,
    }
    canonical_group_total = 0
    lifecycle_event_total = 0
    for source_sha, source_labels in sorted(labels_by_source.items()):
        source_summary = source_summaries[source_sha]
        base_dir = Path(str(source_labels[0].get("_pfill_run_path") or "."))
        source_label_lake_run = label_lake_run or _resolve_path(source_summary.get("label_lake_run"), base_dir)
        source_join_holdout_run = join_holdout_run or _resolve_path(source_summary.get("join_holdout_run"), base_dir)
        if source_label_lake_run is None:
            raise ValueError(f"could not resolve label lake run for source {source_sha}")
        if source_join_holdout_run is None:
            raise ValueError(f"could not resolve join holdout run for source {source_sha}")
        lake_summary, lifecycle_by_node, _, group_index = _load_lifecycle_graph(source_label_lake_run)
        if lake_summary.get("source_telemetry_sha256") != source_sha:
            raise ValueError("label lake and pfill outcome must share source_telemetry_sha256")
        group_facts = _group_facts(lifecycle_by_node, group_index)
        fills_by_group, unresolved_join_reason_counts, join_input_summary = _load_join_fills(
            source_join_holdout_run,
            source_sha,
            group_index,
        )
        for reason, count in unresolved_join_reason_counts.items():
            unresolved_join_reason_counts_total[reason] = unresolved_join_reason_counts_total.get(reason, 0) + count
        cancel_all_by_venue = _load_cancel_all_by_venue(lifecycle_by_node)
        observed_statuses_by_group: dict[str, set[str]] = {}
        for label in source_labels:
            group, _ = _resolve_group(source_sha, label, group_index)
            if group:
                observed_statuses_by_group.setdefault(group, set()).add(str(label.get("outcome_status") or "UNKNOWN"))
        for label in source_labels:
            group, match_key = _resolve_group(source_sha, label, group_index)
            group_fact = group_facts.get(group or "")
            group_fills = fills_by_group.get(group or "", [])
            later_cancel_all = _has_later_cancel_all(label, cancel_all_by_venue)
            audit_status, detail = _audit_status(
                label,
                group_fact=group_fact,
                group_fill_count=len(group_fills),
                group_observed_statuses=observed_statuses_by_group.get(group or "", set()),
                later_cancel_all=later_cancel_all,
            )
            status_counts[audit_status] = status_counts.get(audit_status, 0) + 1
            current_counts["order_label_count"] += 1
            current = _current_status(label)
            if current == "CURRENT_OBSERVED_FILLED":
                current_counts["current_filled_count"] += 1
            elif current == "CURRENT_OBSERVED_NOT_FILLED":
                current_counts["current_not_filled_count"] += 1
            elif current == "CURRENT_CENSORED":
                current_counts["current_censored_count"] += 1
            records.append({
                "schema_version": 1,
                "label_type": "PHASE51E_LIFECYCLE_TRUTH_AUDIT_LABEL",
                "run_id": run_id,
                "baseline_commit": BASELINE_COMMIT,
                "timestamp_local_ns": timestamp_ns + len(records) + 1,
                "source_telemetry_sha256": source_sha,
                "source_pfill_run_id": label.get("run_id"),
                "source_pfill_run_path": label.get("_pfill_run_path"),
                "order_key": label.get("order_key"),
                "order_label_seq": label.get("order_label_seq"),
                "order_source_line": label.get("order_source_line"),
                "order_source_t": label.get("order_source_t"),
                "venue_id": label.get("venue_id"),
                "side": _canonical_side(label.get("side")),
                "current_outcome_status": label.get("outcome_status"),
                "current_p_fill_outcome": label.get("p_fill_outcome"),
                "canonical_group_id": group,
                "canonical_match_key": match_key,
                "canonical_place_count": group_fact.get("place_count") if group_fact else None,
                "canonical_place_intent_count": group_fact.get("place_intent_count") if group_fact else None,
                "canonical_place_ack_count": group_fact.get("place_ack_count") if group_fact else None,
                "canonical_duplicate_place_aliases": group_fact.get("has_duplicate_place_aliases") if group_fact else None,
                "canonical_direct_terminal_count": group_fact.get("direct_terminal_count") if group_fact else None,
                "canonical_replace_terminal_count": group_fact.get("replace_terminal_count") if group_fact else None,
                "canonical_cancel_all_after_order": later_cancel_all,
                "canonical_fill_count": len(group_fills),
                "canonical_status": audit_status,
                "canonical_status_detail": detail,
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
            })
        canonical_group_total += len(group_facts)
        lifecycle_event_total += len(lifecycle_by_node)
        source_input_summaries.append({
            "source_telemetry_sha256": source_sha,
            "label_lake_run": str(source_label_lake_run),
            "label_lake_summary_sha256": _sha256_file(source_label_lake_run / "label_lake_summary.json"),
            "label_lake_labels_sha256": _sha256_file(source_label_lake_run / "labels.jsonl"),
            "join_holdout_run": str(source_join_holdout_run),
            "join_input_summary": join_input_summary,
            "canonical_group_count": len(group_facts),
            "lifecycle_event_count": len(lifecycle_by_node),
        })

    lighter_labels, lighter_input_summary = _load_lighter_gap_labels(lighter_attribution_gap_run, None)
    if lighter_input_summary and lighter_input_summary.get("source_telemetry_sha256") not in source_summaries:
        raise ValueError("lighter attribution gap source_telemetry_sha256 is not present in pfill inputs")
    lighter_records: list[dict[str, Any]] = []
    lighter_reason_counts: dict[str, int] = {}
    for label in lighter_labels:
        reason = str(label.get("gap_reason") or "UNKNOWN")
        lighter_reason_counts[reason] = lighter_reason_counts.get(reason, 0) + 1
        gap_source_sha = str(
            label.get("source_telemetry_sha256")
            or (lighter_input_summary or {}).get("source_telemetry_sha256")
            or ""
        )
        lighter_records.append({
            "schema_version": 1,
            "label_type": "PHASE51E_LIGHTER_NATIVE_IDENTITY_GAP_LABEL",
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "timestamp_local_ns": timestamp_ns + len(lighter_records) + 1,
            "source_telemetry_sha256": gap_source_sha,
            "fill_id": label.get("fill_id"),
            "fill_time_ms": label.get("fill_time_ms"),
            "venue_id": "lighter",
            "side": label.get("side"),
            "price": label.get("price"),
            "size": label.get("size"),
            "observed_maker_taker_role": label.get("observed_maker_taker_role"),
            "native_role_if_determinable": label.get("native_role_if_determinable"),
            "gap_reason": reason,
            "gap_reason_detail": label.get("gap_reason_detail"),
            "order_id_hash_present": label.get("order_id_hash_present"),
            "client_order_id_hash_present": label.get("client_order_id_hash_present"),
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
        })

    raw_lighter_records: list[dict[str, Any]] = []
    raw_lighter_input_summary: dict[str, Any] | None = None
    if lighter_input_summary:
        observed_run = _resolve_path(lighter_input_summary.get("observed_run"), ROOT)
        trade_backfill_run = _resolve_path(lighter_input_summary.get("lighter_trade_backfill_run"), ROOT)
        if observed_run is not None and trade_backfill_run is not None:
            raw_lighter_records, raw_lighter_input_summary = _raw_lighter_native_truth_records(
                observed_run=observed_run,
                lighter_trade_backfill_run=trade_backfill_run,
                run_id=run_id,
                timestamp_ns=timestamp_ns,
                time_tolerance_ms=250,
                price_tolerance=0.000001,
                size_tolerance=0.000001,
            )

    labels_path = out_dir / "order_lifecycle_truth_labels.jsonl"
    lighter_path = out_dir / "lighter_native_identity_gap_labels.jsonl"
    raw_lighter_path = out_dir / "lighter_raw_native_truth_labels.jsonl"
    summary_path = out_dir / "lifecycle_truth_audit_summary.json"
    _write_jsonl(labels_path, records)
    _write_jsonl(lighter_path, lighter_records)
    _write_jsonl(raw_lighter_path, raw_lighter_records)
    gate_reason = (
        "phase51e_canonical_lifecycle_reviewable_movements_found"
        if any(
            status_counts.get(key, 0) > 0
            for key in (
                "CENSORED_TO_CANONICAL_FILLED_REVIEW",
                "CENSORED_TO_CANONICAL_NOT_FILLED_REVIEW",
                "CENSORED_TO_REPLACE_CHAIN_REVIEW",
                "DUPLICATE_PLACE_ALIAS_COLLAPSE_REVIEW",
            )
        )
        else "phase51e_lifecycle_truth_still_unresolved"
    )
    summary = {
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
        "source_telemetry_sha256_list": sorted(source_summaries),
        "source_inputs": source_input_summaries,
        "input_pfill_outcome_runs": pfill_summaries,
        "lighter_attribution_gap_input": lighter_input_summary,
        "lighter_raw_native_truth_input": raw_lighter_input_summary,
        "canonical_group_count": canonical_group_total,
        "lifecycle_event_count": lifecycle_event_total,
        "canonical_status_counts": dict(sorted(status_counts.items())),
        "unresolved_join_reason_counts": dict(sorted(unresolved_join_reason_counts_total.items())),
        "lighter_native_gap_reason_counts": dict(sorted(lighter_reason_counts.items())),
        "lighter_native_gap_label_count": len(lighter_records),
        "lighter_raw_native_truth_label_count": len(raw_lighter_records),
        "lighter_raw_native_match_status_counts": (
            raw_lighter_input_summary.get("raw_native_match_status_counts") if raw_lighter_input_summary else {}
        ),
        "lighter_raw_native_role_counts": (
            raw_lighter_input_summary.get("raw_native_role_counts") if raw_lighter_input_summary else {}
        ),
        "order_lifecycle_truth_labels_sha256": _sha256_file(labels_path),
        "lighter_native_identity_gap_labels_sha256": _sha256_file(lighter_path),
        "lighter_raw_native_truth_labels_sha256": _sha256_file(raw_lighter_path),
        **current_counts,
    }
    _write_json(summary_path, summary)
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, [labels_path, lighter_path, raw_lighter_path, summary_path]),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, [labels_path, lighter_path, raw_lighter_path, summary_path, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pfill-outcome-run", type=Path, action="append", required=True)
    parser.add_argument("--label-lake-run", type=Path, default=None)
    parser.add_argument("--join-holdout-run", type=Path, default=None)
    parser.add_argument("--lighter-attribution-gap-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    args = parser.parse_args()
    try:
        out_dir = build_lifecycle_truth_audit(
            pfill_outcome_runs=args.pfill_outcome_run,
            label_lake_run=args.label_lake_run,
            join_holdout_run=args.join_holdout_run,
            lighter_attribution_gap_run=args.lighter_attribution_gap_run,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
        )
    except Exception as exc:
        print(f"phase51e_lifecycle_truth_audit: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51e_lifecycle_truth_audit: wrote {out_dir}")
    print("phase51e_lifecycle_truth_audit: status HOLD (diagnostic lifecycle/native-truth audit only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
