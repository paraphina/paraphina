#!/usr/bin/env python3
"""Build Phase 5.1y all-venue native-role rows from local source snapshots.

This HOLD-only adapter consumes already-local, sanitized JSON/JSONL snapshots
and emits Phase 5.1v-ready source rows containing explicit venue-native
maker/taker fields plus canonical join keys. It performs no network access,
reads no secrets, submits no orders, and never infers maker/taker role from
strategy intent, post-only status, fee sign, or apparent economics.
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
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51y_all5_native_role_adapter"
VENUES = {"aster", "extended", "hyperliquid", "lighter", "paradex"}
SOURCE_BY_VENUE = {
    "aster": "ASTER_ORDER_TRADE_UPDATE_M",
    "extended": "EXTENDED_ISTAKER",
    "hyperliquid": "HYPERLIQUID_CROSSED",
    "lighter": "LIGHTER_TRADES_JSON",
    "paradex": "PARADEX_LIQUIDITY",
}
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
    "private_key",
    "privatekey",
    "secret_key",
    "secretkey",
    "access_key",
    "accesskey",
    "auth_token",
    "authtoken",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "session_token",
    "signing_key",
}
SOURCE_LIST_KEYS = {
    "data",
    "events",
    "fills",
    "results",
    "rows",
    "trade_history",
    "tradeHistory",
    "trades",
}
ORDER_IDENTITY_FIELDS = {
    "client_order_id",
    "clientOrderId",
    "cloid",
    "oid",
    "order_id",
    "order_id_str",
    "orderId",
    "raw_client_order_id",
    "raw_order_id",
    "trade_id",
    "tid",
    "venue_order_id",
}


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


def _path_text_is_unsafe(path_text: str) -> bool:
    lower = path_text.lower()
    first_part = lower.split("/", 1)[0]
    return lower.startswith(("http://", "https://", "http:/", "https:/")) or "://" in lower or ":" in first_part


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink path is prohibited: {candidate}")


def _check_local_source_path(path: Path) -> Path:
    if _path_text_is_unsafe(str(path)):
        raise ValueError(f"network source paths are prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as native source input: {resolved}")
    _check_no_symlink(resolved)
    if not resolved.exists():
        raise ValueError(f"source path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"source path must be a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"source path must be .json or .jsonl: {resolved}")
    return resolved


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            _check_output_safe(record, path)
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


def _check_no_secret_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_unsafe_flags(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    for obj in _iter_dicts(record):
        leaked = sorted(set(obj) & ORDER_IDENTITY_FIELDS)
        if leaked:
            raise ValueError(f"{path} output leaked raw identifier fields: {leaked}")


def _top_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("canonical_group_id", "order_key", "venue", "venue_id")
        if key in payload
    }


def _payload_records(payload: Any, inherited: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    inherited = dict(inherited or {})
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_payload_records(item, inherited))
        return out
    if not isinstance(payload, dict):
        return []
    inherited_next = {**inherited, **_top_metadata(payload)}
    for key in SOURCE_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            out: list[dict[str, Any]] = []
            for item in value:
                out.extend(_payload_records(item, inherited_next))
            return out
    return [{**inherited, **payload}]


def _parse_source_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        venue_text, path_text = spec.split("=", 1)
        venue = venue_text.strip().lower()
        if venue in VENUES:
            return venue, Path(path_text)
    return None, Path(spec)


def _iter_source_records(specs: list[str]):
    for spec in specs:
        fallback_venue, raw_path = _parse_source_spec(spec)
        source_path = _check_local_source_path(raw_path)
        if source_path.suffix == ".jsonl":
            for line_no, row in _iter_jsonl(source_path):
                _check_unsafe_flags(row, source_path, label="source row")
                _check_no_secret_fields(row, source_path, label="source row")
                for item in _payload_records(row, {"venue_id": fallback_venue} if fallback_venue else None):
                    yield source_path, line_no, fallback_venue, item
            continue
        payload = _load_json(source_path)
        _check_unsafe_flags(payload, source_path, label="source payload")
        _check_no_secret_fields(payload, source_path, label="source payload")
        for line_no, row in enumerate(_payload_records(payload, {"venue_id": fallback_venue} if fallback_venue else None), start=1):
            if isinstance(row, dict):
                _check_unsafe_flags(row, source_path, label="source row")
                _check_no_secret_fields(row, source_path, label="source row")
                yield source_path, line_no, fallback_venue, row


def _base_record(run_id: str, seq: int, timestamp_ns: int, label_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": label_type,
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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


def _load_target_run(target_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_run = _resolve_path(target_run)
    summary_path = target_run / "phase51u_forward_capture_target_manifest_summary.json"
    targets_path = target_run / "native_role_capture_targets.jsonl"
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {summary_path}")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError(f"{summary_path} baseline_commit mismatch")
    _check_unsafe_flags(summary, summary_path, label="target summary")
    targets: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(targets_path):
        _check_unsafe_flags(row, targets_path, label="target row")
        venue = str(row.get("venue_id") or "").lower()
        if venue in VENUES:
            targets.append(row)
    return summary, targets


def _target_maps(targets: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    by_group: dict[str, dict[str, Any]] = {}
    by_order_key: dict[str, set[str]] = {}
    for target in targets:
        group = str(target.get("canonical_group_id") or "")
        if not group:
            raise ValueError("native-role target row missing canonical_group_id")
        if group in by_group:
            raise ValueError(f"duplicate native-role target canonical_group_id: {group}")
        by_group[group] = target
        order_key = str(target.get("order_key") or "")
        if order_key:
            by_order_key.setdefault(order_key, set()).add(group)
    return by_group, by_order_key


def _status_counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _counts_by_venue(targets: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        venue = str(target.get("venue_id") or "unknown").lower()
        counts[venue] = counts.get(venue, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _venue_id(row: dict[str, Any], fallback: str | None) -> str:
    return str(row.get("venue_id") or row.get("venue") or fallback or "").lower()


def _matched_target(
    row: dict[str, Any],
    by_group: dict[str, dict[str, Any]],
    by_order_key: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, str]:
    group = str(row.get("canonical_group_id") or "")
    if group:
        target = by_group.get(group)
        return (target, "SOURCE_ROW_CANONICAL_GROUP_MATCH") if target else (None, "CANONICAL_GROUP_NOT_TARGETED")
    order_key = str(row.get("order_key") or "")
    if order_key:
        groups = by_order_key.get(order_key, set())
        if len(groups) == 1:
            return by_group[next(iter(groups))], "SOURCE_ROW_ORDER_KEY_MATCH"
        if len(groups) > 1:
            return None, "AMBIGUOUS_ORDER_KEY_MATCH"
        return None, "ORDER_KEY_NOT_TARGETED"
    return None, "NO_CANONICAL_LINK"


def _native_fields(row: dict[str, Any], venue: str) -> tuple[dict[str, Any] | None, str]:
    if venue == "aster":
        update = row.get("o") if isinstance(row.get("o"), dict) else row
        maker_flag = update.get("m", update.get("maker_side"))
        fill_qty = update.get("l", update.get("lastFilledQty"))
        if not isinstance(maker_flag, bool):
            return None, "ASTER_MAKER_FLAG_MISSING"
        if not _positive_float(fill_qty):
            return None, "ASTER_POSITIVE_FILL_QTY_MISSING"
        output_update = {"m": maker_flag}
        if update.get("l") not in (None, ""):
            output_update["l"] = update.get("l")
        else:
            output_update["lastFilledQty"] = fill_qty
        out = {"o": output_update, "native_role_source": SOURCE_BY_VENUE[venue]}
        if row.get("e") == "ORDER_TRADE_UPDATE":
            out["e"] = "ORDER_TRADE_UPDATE"
        return out, "ASTER_NATIVE_ROLE_FIELD_EMITTED"
    if venue == "extended":
        is_taker = row.get("isTaker")
        if is_taker is None:
            is_taker = row.get("is_taker")
        if not isinstance(is_taker, bool):
            return None, "EXTENDED_ISTAKER_MISSING"
        return {"isTaker": is_taker, "native_role_source": SOURCE_BY_VENUE[venue]}, "EXTENDED_NATIVE_ROLE_FIELD_EMITTED"
    if venue == "hyperliquid":
        crossed = row.get("crossed")
        if not isinstance(crossed, bool):
            return None, "HYPERLIQUID_CROSSED_MISSING"
        return {"crossed": crossed, "native_role_source": SOURCE_BY_VENUE[venue]}, "HYPERLIQUID_NATIVE_ROLE_FIELD_EMITTED"
    if venue == "lighter":
        account_index = _safe_int(row.get("account_index"))
        is_maker_ask = row.get("is_maker_ask")
        if is_maker_ask is None:
            is_maker_ask = row.get("isMakerAsk")
        ask_account = _safe_int(row.get("ask_account_id") or row.get("askAccountId"))
        bid_account = _safe_int(row.get("bid_account_id") or row.get("bidAccountId"))
        if account_index is None:
            return None, "LIGHTER_ACCOUNT_INDEX_MISSING"
        if not isinstance(is_maker_ask, bool):
            return None, "LIGHTER_IS_MAKER_ASK_MISSING"
        if ask_account is None or bid_account is None:
            return None, "LIGHTER_SIDE_ACCOUNT_ID_MISSING"
        return {
            "account_index": account_index,
            "is_maker_ask": is_maker_ask,
            "ask_account_id": ask_account,
            "bid_account_id": bid_account,
            "native_role_source": SOURCE_BY_VENUE[venue],
        }, "LIGHTER_NATIVE_ROLE_FIELD_EMITTED"
    if venue == "paradex":
        liquidity = str(row.get("liquidity") or "").upper()
        if liquidity not in {"MAKER", "TAKER"}:
            return None, "PARADEX_LIQUIDITY_MISSING"
        return {"liquidity": liquidity, "native_role_source": SOURCE_BY_VENUE[venue]}, "PARADEX_NATIVE_ROLE_FIELD_EMITTED"
    return None, "UNSUPPORTED_VENUE"


def build_all5_native_role_adapter(
    *,
    target_run: Path,
    source_json: list[str],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
) -> Path:
    if not source_json:
        raise ValueError("at least one --source-json is required")
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    target_summary, targets = _load_target_run(target_run)
    by_group, by_order_key = _target_maps(targets)
    target_groups = set(by_group)
    recovered_groups_by_venue: dict[str, set[str]] = {venue: set() for venue in VENUES}
    emitted_source_hashes: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    source_artifacts: dict[str, dict[str, Any]] = {}
    source_rows_seen = 0

    for seq, (path, line_no, fallback_venue, row) in enumerate(_iter_source_records(source_json), start=1):
        source_rows_seen += 1
        source_path_hash = _stable_hash(str(path))
        artifact = source_artifacts.setdefault(
            source_path_hash,
            {"path_hash": source_path_hash, "sha256": _sha256_file(path), "source_row_count": 0},
        )
        artifact["source_row_count"] += 1
        source_hash = _stable_hash(row)
        target, join_status = _matched_target(row, by_group, by_order_key)
        venue = _venue_id(row, fallback_venue)
        status = join_status
        native_payload: dict[str, Any] | None = None
        if target is not None:
            expected_venue = str(target.get("venue_id") or "").lower()
            if venue != expected_venue:
                status = "TARGET_VENUE_MISMATCH"
            else:
                native_payload, native_status = _native_fields(row, venue)
                status = native_status

        if target is not None and native_payload is not None and source_hash not in emitted_source_hashes:
            group = str(target.get("canonical_group_id") or "")
            output = _base_record(run_id, len(output_rows), timestamp_ns, "PHASE51Y_ALL5_NATIVE_ROLE_SOURCE")
            output.update(
                {
                    "venue_id": venue,
                    "canonical_group_id": group,
                    "order_key": target.get("order_key"),
                    "source_record_sha256": source_hash,
                    **native_payload,
                }
            )
            output_rows.append(output)
            emitted_source_hashes.add(source_hash)
            recovered_groups_by_venue.setdefault(venue, set()).add(group)

        label = _base_record(run_id, seq, timestamp_ns, "PHASE51Y_ALL5_NATIVE_ROLE_ADAPTER_LABEL")
        label.update(
            {
                "source_path_hash": source_path_hash,
                "source_line": line_no,
                "source_record_sha256": source_hash,
                "venue_id": venue or "unknown",
                "fallback_venue_id": fallback_venue,
                "canonical_group_id": str(target.get("canonical_group_id") or "") if target else None,
                "order_key": str(target.get("order_key") or "") if target else None,
                "target_join_status": join_status,
                "adapter_status": status,
                "source_fields_preserved": sorted(native_payload) if native_payload else [],
            }
        )
        labels.append(label)

    output_path = out_dir / "all5_forward_native_role_snapshot.jsonl"
    labels_path = out_dir / "all5_native_role_adapter_labels.jsonl"
    summary_path = out_dir / "phase51y_all5_native_role_adapter_summary.json"
    manifest_path = out_dir / "phase51y_manifest.json"
    _write_jsonl(output_path, output_rows)
    _write_jsonl(labels_path, labels)

    recovered_groups = {str(row.get("canonical_group_id") or "") for row in output_rows}
    recovered_by_venue = {venue: len(groups) for venue, groups in sorted(recovered_groups_by_venue.items()) if groups}
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": (
            "phase51y_all5_native_role_adapter_complete_nonlive_hold"
            if target_groups and target_groups <= recovered_groups
            else "phase51y_all5_native_role_adapter_incomplete_nonlive_hold"
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
        "raw_identifier_redaction_status": "PASS",
        "target_run": str(_resolve_path(target_run)),
        "target_gate_status": target_summary.get("gate_status"),
        "native_role_target_count": len(targets),
        "native_role_target_counts_by_venue": _counts_by_venue(targets),
        "native_role_target_recovered_count": len(target_groups & recovered_groups),
        "native_role_target_recovered_counts_by_venue": recovered_by_venue,
        "source_file_count": len(source_json),
        "source_row_count": source_rows_seen,
        "source_row_emitted_count": len(output_rows),
        "adapter_status_counts": _status_counts(labels, "adapter_status"),
        "output_path": str(output_path),
        "clears_phase51_blockers": False,
        "source_artifacts": sorted(source_artifacts.values(), key=lambda item: item["path_hash"]),
    }
    _write_json(summary_path, summary)
    artifacts = [output_path, labels_path, summary_path]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": _artifact_infos(out_dir, artifacts),
        },
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument(
        "--source-json",
        action="append",
        default=[],
        help="Local .json/.jsonl source path, optionally prefixed as venue=/path.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51y_all5_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_all5_native_role_adapter(
            target_run=args.target_run,
            source_json=args.source_json,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51y_all5_native_role_adapter: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51y_all5_native_role_adapter: wrote {out_dir}")
    print("phase51y_all5_native_role_adapter: status HOLD (native role source only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
